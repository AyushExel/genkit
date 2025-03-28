/**
 * Copyright 2024 Google LLC
 * Copyright 2024 LanceDB (modified for LanceDB)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

import {
  connect, // Main lancedb connection function
  Connection,
  Table,
  WriteMode, // Use WriteMode enum for clarity
} from '@lancedb/lancedb'; // LanceDB Typescript SDK
import { Genkit, z } from 'genkit'; // Assuming genkit core exists
import { GenkitPlugin, genkitPlugin } from 'genkit/plugin'; // Assuming genkit plugin helpers exist

import { EmbedderArgument, Embedding } from 'genkit/embedder'; // Assuming genkit embedder types
import {
  CommonRetrieverOptionsSchema,
  Document,
  indexerRef,
  retrieverRef,
} from 'genkit/retriever'; // Assuming genkit retriever types
import { Md5 } from 'ts-md5'; // For generating IDs
import * as arrow from 'apache-arrow'; // LanceDB often works with Arrow data types

// --- Configuration Schemas ---

// Removed SparseVectorSchema as it's Pinecone-specific

const LanceDBRetrieverOptionsSchema = CommonRetrieverOptionsSchema.extend({
  k: z.number().int().positive().default(10), // Default k value
  whereFilter: z.string().optional(), // SQL-like filter string for LanceDB
  vectorColumnName: z.string().default('vector'), // Default vector column name
  // Note: LanceDB search typically returns metadata; includeMetadata=True is implicit
  //       includeValues is usually False for retrieval, vectors fetched separately if needed.
});

const LanceDBIndexerOptionsSchema = z.object({
  dbUri: z.string(), // LanceDB connection URI (e.g., "./.lancedb", "db://my_database")
  tableName: z.string(),
  vectorColumnName: z.string().default('vector'),
  textColumnName: z.string().default('text'), // Column to store document content
  metadataColumnName: z.string().default('metadata'), // Column to store stringified JSON metadata
  writeMode: z.nativeEnum(WriteMode).default(WriteMode.Append), // Append or Overwrite
});

// Define the expected data structure for LanceDB add/create
// Adjust based on actual column names used
interface LanceDbDataRow {
  id: string; // Unique ID for the vector chunk
  [vectorColumn: string]: number[]; // Dynamically named vector column
  [textColumn: string]: string; // Dynamically named text column
  [metadataColumn: string]: string; // Dynamically named metadata column (JSON string)
  // Add other metadata fields directly if preferred over a single JSON string column
}

// --- Ref Helpers ---

/**
 * lancedbRetrieverRef function creates a retriever reference for LanceDB.
 * @param params The params for the new LanceDB retriever
 * @param params.tableName The tableName for the LanceDB retriever
 * @param params.displayName A display name for the retriever. If not specified, defaults to `LanceDB - <tableName>`
 * @returns A reference to a LanceDB retriever.
 */
export const lancedbRetrieverRef = (params: {
  tableName: string;
  displayName?: string;
}) => {
  return retrieverRef({
    name: `lancedb/${params.tableName}`,
    info: {
      label: params.displayName ?? `LanceDB - ${params.tableName}`,
    },
    configSchema: LanceDBRetrieverOptionsSchema,
  });
};

/**
 * lancedbIndexerRef function creates an indexer reference for LanceDB.
 * @param params The params for the new LanceDB indexer.
 * @param params.tableName The tableName for the LanceDB indexer.
 * @param params.displayName A display name for the indexer. If not specified, defaults to `LanceDB - <tableName>`
 * @returns A reference to a LanceDB indexer.
 */
export const lancedbIndexerRef = (params: {
  tableName: string;
  displayName?: string;
}) => {
  return indexerRef({
    name: `lancedb/${params.tableName}`,
    info: {
      label: params.displayName ?? `LanceDB - ${params.tableName}`,
    },
    configSchema: LanceDBIndexerOptionsSchema,
  });
};

// --- Plugin Definition ---

/**
 * LanceDB plugin that provides a LanceDB retriever and indexer.
 * @param configs An array of configurations, each containing:
 * dbUri: The LanceDB connection URI (e.g., "./.lancedb").
 * tableName: The name of the table for this indexer/retriever.
 * embedder: The embedder to use.
 * embedderOptions: Optional options for the embedder.
 * vectorColumnName: (Optional) Name of the vector column (default: "vector").
 * textColumnName: (Optional) Name of the text content column (default: "text").
 * metadataColumnName: (Optional) Name of the metadata column (default: "metadata").
 * @returns The LanceDB Genkit plugin.
 */
export function lancedb<EmbedderCustomOptions extends z.ZodTypeAny>(
  configs: {
    dbUri: string;
    tableName: string;
    embedder: EmbedderArgument<EmbedderCustomOptions>;
    embedderOptions?: z.infer<EmbedderCustomOptions>;
    vectorColumnName?: string;
    textColumnName?: string;
    metadataColumnName?: string;
  }[]
): GenkitPlugin {
  return genkitPlugin('lancedb', async (ai: Genkit) => {
    configs.forEach((config) => {
      configureLanceDBRetriever(ai, config);
      configureLanceDBIndexer(ai, config);
    });
  });
}

export default lancedb; // Default export for convenience

// --- Retriever Configuration ---

export function configureLanceDBRetriever<
  EmbedderCustomOptions extends z.ZodTypeAny,
>(
  ai: Genkit,
  params: {
    dbUri: string;
    tableName: string;
    embedder: EmbedderArgument<EmbedderCustomOptions>;
    embedderOptions?: z.infer<EmbedderCustomOptions>;
    vectorColumnName?: string;
    textColumnName?: string;
    metadataColumnName?: string;
  }
) {
  const {
    dbUri,
    tableName,
    embedder,
    embedderOptions,
    vectorColumnName = 'vector', // Use defaults
    textColumnName = 'text',
    metadataColumnName = 'metadata',
  } = params;

  // Define the retriever
  ai.defineRetriever(
    {
      name: `lancedb/${tableName}`,
      configSchema: LanceDBRetrieverOptionsSchema,
    },
    async (
      content: Document, // Changed from 'content' string to Document type
      options: z.infer<typeof LanceDBRetrieverOptionsSchema>
    ): Promise<{ documents: Document[] }> => {
      let db: Connection | null = null;
      let tbl: Table | null = null;
      try {
        db = await connect(dbUri);
        tbl = await db.openTable(tableName);
      } catch (error: any) {
        console.error(
          `LanceDB Retriever Error: Failed to connect or open table '${tableName}' at '${dbUri}'. Does it exist?`,
          error
        );
        // Depending on requirements, might re-throw or return empty results
        return { documents: [] };
      }

      // 1. Embed the query content
      // Assuming ai.embed takes a Document and returns embeddings for its content parts
      const queryEmbeddings: Embedding[] = await ai.embed({
        embedder,
        content: content, // Pass the Document object
        options: embedderOptions,
      });

      if (!queryEmbeddings || queryEmbeddings.length === 0) {
        console.warn('LanceDB Retriever: Query embedding resulted in no vectors.');
        return { documents: [] };
      }
      const queryVector = queryEmbeddings[0].embedding; // Use the first embedding for query

      // 2. Perform LanceDB search
      let searchQuery = tbl
        .vectorSearch(queryVector, vectorColumnName) // Use vectorSearch method
        .limit(options.k);

      if (options.whereFilter) {
        searchQuery = searchQuery.where(options.whereFilter);
      }

      // Select columns needed to reconstruct the Document
      const selectColumns = Array.from(
        new Set([textColumnName, metadataColumnName])
      ); // Ensure unique
      searchQuery = searchQuery.select(selectColumns);

      try {
        const results: Record<string, any>[] = await searchQuery.toArray(); // Execute query

        // 3. Format results as Genkit Documents
        const documents: Document[] = results.map((res) => {
          const docContent = res[textColumnName] ?? '';
          const metadataStr = res[metadataColumnName] ?? '{}';
          let docMetadata: Record<string, unknown> = {};

          try {
            docMetadata = JSON.parse(metadataStr);
          } catch (e) {
            console.warn(
              `LanceDB Retriever: Failed to parse metadata for a result from table '${tableName}'. Content: "${metadataStr}"`,
              e
            );
            docMetadata = { error: 'Failed to parse metadata' };
          }

          // Assuming Document.fromData exists or create Document instances directly
          // Adjust based on actual Genkit Document structure
          return Document.fromData(docContent, undefined, docMetadata);
        });

        return { documents };
      } catch (searchError: any) {
        console.error(
          `LanceDB Retriever Error: Search failed for table '${tableName}'`,
          searchError
        );
        return { documents: [] }; // Return empty on search failure
      }
    }
  );
}

// --- Indexer Configuration ---

export function configureLanceDBIndexer<
  EmbedderCustomOptions extends z.ZodTypeAny,
>(
  ai: Genkit,
  params: {
    dbUri: string;
    tableName: string;
    embedder: EmbedderArgument<EmbedderCustomOptions>;
    embedderOptions?: z.infer<EmbedderCustomOptions>;
    vectorColumnName?: string;
    textColumnName?: string;
    metadataColumnName?: string;
  }
) {
  const {
    dbUri,
    tableName,
    embedder,
    embedderOptions,
    vectorColumnName = 'vector', // Use defaults
    textColumnName = 'text',
    metadataColumnName = 'metadata',
  } = params;

  ai.defineIndexer(
    {
      name: `lancedb/${tableName}`,
      configSchema: LanceDBIndexerOptionsSchema,
    },
    async (
      docs: Document[],
      options?: z.infer<typeof LanceDBIndexerOptionsSchema> // Passed options override defaults
    ) => {
      const effectiveDbUri = options?.dbUri ?? dbUri;
      const effectiveTableName = options?.tableName ?? tableName;
      const effectiveVectorCol = options?.vectorColumnName ?? vectorColumnName;
      const effectiveTextCol = options?.textColumnName ?? textColumnName;
      const effectiveMetadataCol = options?.metadataColumnName ?? metadataColumnName;
      const effectiveWriteMode = options?.writeMode ?? WriteMode.Append;

      if (!docs || docs.length === 0) {
        console.log('LanceDB Indexer: No documents provided to index.');
        return;
      }

      let db: Connection;
      try {
        db = await connect(effectiveDbUri);
      } catch (error: any) {
        console.error(
          `LanceDB Indexer Error: Failed to connect to '${effectiveDbUri}'`,
          error
        );
        throw new Error(
          `LanceDB Indexer: Connection failed to ${effectiveDbUri}`
        );
      }

      let tableExists = true;
      let tbl: Table | null = null;
      try {
        tbl = await db.openTable(effectiveTableName);
        console.log(
          `LanceDB Indexer: Using existing table '${effectiveTableName}'. Mode: ${effectiveWriteMode}`
        );
        if (effectiveWriteMode === WriteMode.Overwrite) {
          console.warn(
            `LanceDB Indexer: Overwriting existing table '${effectiveTableName}'!`
          );
          // Overwrite happens during the add/create call if mode is Overwrite
        }
      } catch (e) {
        tableExists = false;
        console.log(
          `LanceDB Indexer: Table '${effectiveTableName}' not found. Will attempt to create.`
        );
      }

      // 1. Embed all documents concurrently
      const embeddingPromises = docs.map((doc) =>
        ai.embed({
          embedder,
          content: doc, // Embed the document
          options: embedderOptions,
        })
      );
      const embeddingsForEachDoc: Embedding[][] = await Promise.all(
        embeddingPromises
      );

      // 2. Prepare data for LanceDB add/create
      const dataToAdd: LanceDbDataRow[] = [];
      docs.forEach((doc, i) => {
        const docEmbeddings: Embedding[] = embeddingsForEachDoc[i];

        // Create one LanceDB row per embedding chunk
        docEmbeddings.forEach((embedding) => {
          // Generate an ID based on embedding content and metadata
          // Using Md5 like the Pinecone example
          const docRepr = `${embedding.content}-${JSON.stringify(doc.metadata || {})}`;
          const id = Md5.hashStr(docRepr);

          // Ensure metadata is stringified safely
          let metadataJson = '{}';
          try {
            metadataJson = JSON.stringify(doc.metadata || {});
          } catch (jsonError) {
            console.warn(
              `LanceDB Indexer: Could not stringify metadata for doc ${id}`,
              jsonError
            );
          }

          const row: LanceDbDataRow = {
            id: id,
            [effectiveVectorCol]: embedding.embedding,
            [effectiveTextCol]: embedding.content, // Use content from embedding chunk
            [effectiveMetadataCol]: metadataJson,
          };
          dataToAdd.push(row);
        });
      });

      if (dataToAdd.length === 0) {
        console.log('LanceDB Indexer: No data rows generated after embedding.');
        return;
      }

      // 3. Add data to LanceDB (create table if needed)
      try {
        if (!tableExists) {
          // Create table with the first batch
          console.log(
            `LanceDB Indexer: Creating table '${effectiveTableName}' with ${dataToAdd.length} records.`
          );
          // LanceDB infers schema from the data
          tbl = await db.createTable(effectiveTableName, dataToAdd, {
            mode: WriteMode.Create, // Explicitly create
          });
          console.log(
            `LanceDB Indexer: Successfully created table '${effectiveTableName}'.`
          );
        } else if (tbl) {
          // Add data to existing table
          if (effectiveWriteMode === WriteMode.Overwrite) {
            console.log(
              `LanceDB Indexer: Overwriting table '${effectiveTableName}' with ${dataToAdd.length} records.`
            );
            // Create table with overwrite mode effectively replaces the table
            tbl = await db.createTable(effectiveTableName, dataToAdd, {
              mode: WriteMode.Overwrite,
            });
          } else {
            console.log(
              `LanceDB Indexer: Appending ${dataToAdd.length} records to table '${effectiveTableName}'.`
            );
            await tbl.add(dataToAdd);
          }
          console.log(
            `LanceDB Indexer: Successfully added data to table '${effectiveTableName}'.`
          );
        }
      } catch (writeError: any) {
        console.error(
          `LanceDB Indexer Error: Failed to write data to table '${effectiveTableName}'`,
          writeError
        );
        // Rethrow or handle as appropriate for your application
        throw new Error(
          `LanceDB Indexer: Failed writing to table ${effectiveTableName}`
        );
      }
    }
  );
}

// --- Optional Helper Functions ---

/**
 * Helper function for creating a LanceDB table explicitly.
 * Primarily useful if you need to create an empty table or ensure specific settings.
 * @param params Parameters for table creation
 * @param params.dbUri Connection URI
 * @param params.tableName Name of the table to create
 * @param params.exampleData Optional example data (list of objects) to infer schema.
 * LanceDB TS SDK primarily uses data inference. Defining schema
 * programmatically might require Arrow schema objects.
 */
export async function createLanceDBTable(params: {
  dbUri: string;
  tableName: string;
  exampleData?: Record<string, any>[]; // Data to infer schema
}): Promise<Table> {
  const { dbUri, tableName, exampleData } = params;
  if (!exampleData || exampleData.length === 0) {
    throw new Error(
      'LanceDB Helper: Example data must be provided to infer schema for table creation in TS SDK.'
    );
  }
  const db = await connect(dbUri);
  console.log(`LanceDB Helper: Creating table '${tableName}' at '${dbUri}'...`);
  const tbl = await db.createTable(tableName, exampleData, {
    mode: WriteMode.Create,
  });
  console.log(`LanceDB Helper: Table '${tableName}' created successfully.`);
  return tbl;
}

/**
 * Helper function for deleting a LanceDB table.
 * @param params Parameters for deletion
 * @param params.dbUri Connection URI
 * @param params.tableName Name of the table to delete
 */
export async function deleteLanceDBTable(params: {
  dbUri: string;
  tableName: string;
}): Promise<void> {
  const { dbUri, tableName } = params;
  const db = await connect(dbUri);
  console.log(`LanceDB Helper: Deleting table '${tableName}' from '${dbUri}'...`);
  await db.dropTable(tableName);
  console.log(`LanceDB Helper: Table '${tableName}' deleted successfully.`);
}