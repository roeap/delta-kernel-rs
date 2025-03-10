//! Default Json handler implementation

use std::io::BufReader;
use std::ops::Range;
use std::sync::Arc;
use std::task::{ready, Poll};

use crate::arrow::datatypes::SchemaRef as ArrowSchemaRef;
use crate::arrow::json::ReaderBuilder;
use bytes::{Buf, Bytes};
use futures::{StreamExt, TryStreamExt};
use object_store::path::Path;
use object_store::{DynObjectStore, GetResultPayload};
use url::Url;

use super::executor::TaskExecutor;
use super::file_stream::{FileOpenFuture, FileOpener, FileStream};
use crate::engine::arrow_utils::parse_json as arrow_parse_json;
use crate::engine::arrow_utils::to_json_bytes;
use crate::schema::SchemaRef;
use crate::{
    DeltaResult, EngineData, Error, ExpressionRef, FileDataReadResultIterator, FileMeta,
    JsonHandler,
};

#[derive(Debug)]
pub struct DefaultJsonHandler<E: TaskExecutor> {
    /// The object store to read files from
    store: Arc<DynObjectStore>,
    /// The executor to run async tasks on
    task_executor: Arc<E>,
    /// The maximum number of batches to read ahead
    readahead: usize,
    /// The number of rows to read per batch
    batch_size: usize,
}

impl<E: TaskExecutor> DefaultJsonHandler<E> {
    pub fn new(store: Arc<DynObjectStore>, task_executor: Arc<E>) -> Self {
        Self {
            store,
            task_executor,
            readahead: 10,
            batch_size: 1024,
        }
    }

    /// Set the maximum number of batches to read ahead during [Self::read_json_files()].
    ///
    /// Defaults to 10.
    pub fn with_readahead(mut self, readahead: usize) -> Self {
        self.readahead = readahead;
        self
    }

    /// Set the number of rows to read per batch during [Self::parse_json()].
    ///
    /// Defaults to 1024.
    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size;
        self
    }
}

impl<E: TaskExecutor> JsonHandler for DefaultJsonHandler<E> {
    fn parse_json(
        &self,
        json_strings: Box<dyn EngineData>,
        output_schema: SchemaRef,
    ) -> DeltaResult<Box<dyn EngineData>> {
        arrow_parse_json(json_strings, output_schema)
    }

    fn read_json_files(
        &self,
        files: &[FileMeta],
        physical_schema: SchemaRef,
        _predicate: Option<ExpressionRef>,
    ) -> DeltaResult<FileDataReadResultIterator> {
        if files.is_empty() {
            return Ok(Box::new(std::iter::empty()));
        }

        let schema: ArrowSchemaRef = Arc::new(physical_schema.as_ref().try_into()?);
        let file_opener = JsonOpener::new(self.batch_size, schema.clone(), self.store.clone());
        FileStream::new_async_read_iterator(
            self.task_executor.clone(),
            schema,
            Box::new(file_opener),
            files,
            self.readahead,
        )
    }

    // note: for now we just buffer all the data and write it out all at once
    fn write_json_file(
        &self,
        path: &Url,
        data: Box<dyn Iterator<Item = DeltaResult<Box<dyn EngineData>>> + Send + '_>,
        _overwrite: bool,
    ) -> DeltaResult<()> {
        let buffer = to_json_bytes(data)?;
        // Put if absent
        let store = self.store.clone(); // cheap Arc
        let path = Path::from(path.path());
        let path_str = path.to_string();
        self.task_executor
            .block_on(async move {
                store
                    .put_opts(&path, buffer.into(), object_store::PutMode::Create.into())
                    .await
            })
            .map_err(|e| match e {
                object_store::Error::AlreadyExists { .. } => Error::FileAlreadyExists(path_str),
                e => e.into(),
            })?;
        Ok(())
    }
}

/// A [`FileOpener`] that opens a JSON file and yields a [`FileOpenFuture`]
#[allow(missing_debug_implementations)]
pub struct JsonOpener {
    batch_size: usize,
    projected_schema: ArrowSchemaRef,
    object_store: Arc<DynObjectStore>,
}

impl JsonOpener {
    /// Returns a  [`JsonOpener`]
    pub fn new(
        batch_size: usize,
        projected_schema: ArrowSchemaRef,
        // file_compression_type: FileCompressionType,
        object_store: Arc<DynObjectStore>,
    ) -> Self {
        Self {
            batch_size,
            projected_schema,
            // file_compression_type,
            object_store,
        }
    }
}

impl FileOpener for JsonOpener {
    fn open(&self, file_meta: FileMeta, _: Option<Range<i64>>) -> DeltaResult<FileOpenFuture> {
        let store = self.object_store.clone();
        let schema = self.projected_schema.clone();
        let batch_size = self.batch_size;

        Ok(Box::pin(async move {
            let path = Path::from_url_path(file_meta.location.path())?;
            match store.get(&path).await?.payload {
                GetResultPayload::File(file, _) => {
                    let reader = ReaderBuilder::new(schema)
                        .with_batch_size(batch_size)
                        .build(BufReader::new(file))?;
                    Ok(futures::stream::iter(reader).map_err(Error::from).boxed())
                }
                GetResultPayload::Stream(s) => {
                    let mut decoder = ReaderBuilder::new(schema)
                        .with_batch_size(batch_size)
                        .build_decoder()?;

                    let mut input = s.map_err(Error::from);
                    let mut buffered = Bytes::new();

                    let s = futures::stream::poll_fn(move |cx| {
                        loop {
                            if buffered.is_empty() {
                                buffered = match ready!(input.poll_next_unpin(cx)) {
                                    Some(Ok(b)) => b,
                                    Some(Err(e)) => return Poll::Ready(Some(Err(e))),
                                    None => break,
                                };
                            }
                            let read = buffered.len();

                            let decoded = match decoder.decode(buffered.as_ref()) {
                                Ok(decoded) => decoded,
                                Err(e) => return Poll::Ready(Some(Err(e.into()))),
                            };

                            buffered.advance(decoded);
                            if decoded != read {
                                break;
                            }
                        }

                        Poll::Ready(decoder.flush().map_err(Error::from).transpose())
                    });
                    Ok(s.map_err(Error::from).boxed())
                }
            }
        }))
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use crate::arrow::array::{AsArray, RecordBatch, StringArray};
    use crate::arrow::datatypes::{DataType, Field, Schema as ArrowSchema};
    use itertools::Itertools;
    use object_store::{local::LocalFileSystem, ObjectStore};

    use super::*;
    use crate::{
        actions::get_log_schema, engine::arrow_data::ArrowEngineData,
        engine::default::executor::tokio::TokioBackgroundExecutor,
    };

    fn string_array_to_engine_data(string_array: StringArray) -> Box<dyn EngineData> {
        let string_field = Arc::new(Field::new("a", DataType::Utf8, true));
        let schema = Arc::new(ArrowSchema::new(vec![string_field]));
        let batch = RecordBatch::try_new(schema, vec![Arc::new(string_array)])
            .expect("Can't convert to record batch");
        Box::new(ArrowEngineData::new(batch))
    }

    #[test]
    fn test_parse_json() {
        let store = Arc::new(LocalFileSystem::new());
        let handler = DefaultJsonHandler::new(store, Arc::new(TokioBackgroundExecutor::new()));

        let json_strings = StringArray::from(vec![
            r#"{"add":{"path":"part-00000-fae5310a-a37d-4e51-827b-c3d5516560ca-c000.snappy.parquet","partitionValues":{},"size":635,"modificationTime":1677811178336,"dataChange":true,"stats":"{\"numRecords\":10,\"minValues\":{\"value\":0},\"maxValues\":{\"value\":9},\"nullCount\":{\"value\":0},\"tightBounds\":true}","tags":{"INSERTION_TIME":"1677811178336000","MIN_INSERTION_TIME":"1677811178336000","MAX_INSERTION_TIME":"1677811178336000","OPTIMIZE_TARGET_SIZE":"268435456"}}}"#,
            r#"{"commitInfo":{"timestamp":1677811178585,"operation":"WRITE","operationParameters":{"mode":"ErrorIfExists","partitionBy":"[]"},"isolationLevel":"WriteSerializable","isBlindAppend":true,"operationMetrics":{"numFiles":"1","numOutputRows":"10","numOutputBytes":"635"},"engineInfo":"Databricks-Runtime/<unknown>","txnId":"a6a94671-55ef-450e-9546-b8465b9147de"}}"#,
            r#"{"protocol":{"minReaderVersion":3,"minWriterVersion":7,"readerFeatures":["deletionVectors"],"writerFeatures":["deletionVectors"]}}"#,
            r#"{"metaData":{"id":"testId","format":{"provider":"parquet","options":{}},"schemaString":"{\"type\":\"struct\",\"fields\":[{\"name\":\"value\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}}]}","partitionColumns":[],"configuration":{"delta.enableDeletionVectors":"true","delta.columnMapping.mode":"none"},"createdTime":1677811175819}}"#,
        ]);
        let output_schema = get_log_schema().clone();

        let batch = handler
            .parse_json(string_array_to_engine_data(json_strings), output_schema)
            .unwrap();
        assert_eq!(batch.len(), 4);
    }

    #[test]
    fn test_parse_json_drop_field() {
        let store = Arc::new(LocalFileSystem::new());
        let handler = DefaultJsonHandler::new(store, Arc::new(TokioBackgroundExecutor::new()));
        let json_strings = StringArray::from(vec![
            r#"{"add":{"path":"part-00000-fae5310a-a37d-4e51-827b-c3d5516560ca-c000.snappy.parquet","partitionValues":{},"size":635,"modificationTime":1677811178336,"dataChange":true,"stats":"{\"numRecords\":10,\"minValues\":{\"value\":0},\"maxValues\":{\"value\":9},\"nullCount\":{\"value\":0},\"tightBounds\":false}","tags":{"INSERTION_TIME":"1677811178336000","MIN_INSERTION_TIME":"1677811178336000","MAX_INSERTION_TIME":"1677811178336000","OPTIMIZE_TARGET_SIZE":"268435456"},"deletionVector":{"storageType":"u","pathOrInlineDv":"vBn[lx{q8@P<9BNH/isA","offset":1,"sizeInBytes":36,"cardinality":2, "maxRowId": 3}}}"#,
        ]);
        let output_schema = get_log_schema().clone();

        let batch: RecordBatch = handler
            .parse_json(string_array_to_engine_data(json_strings), output_schema)
            .unwrap()
            .into_any()
            .downcast::<ArrowEngineData>()
            .map(|sd| sd.into())
            .unwrap();
        assert_eq!(batch.column(0).len(), 1);
        let add_array = batch.column_by_name("add").unwrap().as_struct();
        let dv_col = add_array
            .column_by_name("deletionVector")
            .unwrap()
            .as_struct();
        assert!(dv_col.column_by_name("storageType").is_some());
        assert!(dv_col.column_by_name("maxRowId").is_none());
    }

    #[tokio::test]
    async fn test_read_json_files() {
        let store = Arc::new(LocalFileSystem::new());

        let path = std::fs::canonicalize(PathBuf::from(
            "./tests/data/table-with-dv-small/_delta_log/00000000000000000000.json",
        ))
        .unwrap();
        let url = url::Url::from_file_path(path).unwrap();
        let location = Path::from(url.path());
        let meta = store.head(&location).await.unwrap();

        let files = &[FileMeta {
            location: url.clone(),
            last_modified: meta.last_modified.timestamp_millis(),
            size: meta.size,
        }];

        let handler = DefaultJsonHandler::new(store, Arc::new(TokioBackgroundExecutor::new()));
        let physical_schema = Arc::new(ArrowSchema::try_from(get_log_schema().as_ref()).unwrap());
        let data: Vec<RecordBatch> = handler
            .read_json_files(files, Arc::new(physical_schema.try_into().unwrap()), None)
            .unwrap()
            .map(|ed_res| {
                // TODO(nick) make this easier
                ed_res.and_then(|ed| {
                    ed.into_any()
                        .downcast::<ArrowEngineData>()
                        .map_err(|_| Error::engine_data_type("ArrowEngineData"))
                        .map(|sd| sd.into())
                })
            })
            .try_collect()
            .unwrap();

        assert_eq!(data.len(), 1);
        assert_eq!(data[0].num_rows(), 4);
    }
}
