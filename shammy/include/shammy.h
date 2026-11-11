/* Shammy: C interface for reading .shammy vector files.
 *
 * Link against shammy.dll (Windows), libshammy.so (Linux) or libshammy.dylib (macOS), built by
 *   cargo build --release -p shammy-ffi
 *
 * Rules
 *   - Every function returns SHAMMY_OK (0) or a negative SHAMMY_E_* code. After a failure,
 *     shammy_last_error() gives the message for the calling thread.
 *   - Strings are UTF-8 and NUL-terminated.
 *   - A reader may be used from many threads at once. Close it once, when no call is running.
 *   - A reader sees the file as of its last commit when it was opened. Open a new reader to see
 *     later commits. Replacing the file needs every reader on it closed first (Windows).
 *   - Scores are cosine similarity. Results are the same on every CPU and for any thread count.
 */
#ifndef SHAMMY_H
#define SHAMMY_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define SHAMMY_OK          0
#define SHAMMY_E_ARGUMENT -1  /* a bad argument: null pointer, wrong query size, unknown id */
#define SHAMMY_E_IO       -2  /* the file can't be opened or read */
#define SHAMMY_E_FORMAT   -3  /* not a Shammy file, a newer format, or damaged */
#define SHAMMY_E_INTERNAL -4  /* a bug in the library */

typedef struct ShammyReader ShammyReader;

typedef struct ShammyInfo {
    uint32_t dims;             /* stored values per vector */
    uint32_t sketch_dims;
    uint32_t lossless;         /* 1 when float16 copies are stored (exact rescoring, shammy_vector) */
    uint32_t segments;
    uint32_t file_type_count;  /* names via shammy_file_type */
    uint32_t reserved;
    uint64_t generation;       /* commits so far */
    uint64_t rows;             /* including deleted rows */
    uint64_t live_rows;
    uint64_t file_bytes;
    uint64_t instruction_hash; /* XXH3-64 of the query instruction the vectors were built with */
    char     model[128];       /* embedding model name */
} ShammyInfo;

typedef struct ShammyHit {
    uint64_t id;      /* chunk id */
    uint32_t doc_id;
    float    score;   /* cosine similarity */
} ShammyHit;

/* All zero means defaults. */
typedef struct ShammyParams {
    uint32_t threads;  /* 0 = shammy_default_threads() */
    uint32_t exact;    /* 1 = score every row in full (slow; needs a lossless file) */
    uint64_t stage1;   /* rows kept by the sketch scan; 0 = 1% of live rows, at least 4096 */
    uint64_t stage2;   /* rows kept by the 1-bit rerank; 0 = 256 */
    uint64_t finish;   /* rows rescored exactly; 0 = 50 */
} ShammyParams;

typedef struct ShammyReport {
    uint32_t ok;                /* 1 when every checksum matches */
    uint32_t problems;
    uint64_t generation;
    uint64_t rows;
    uint64_t live_rows;
    uint64_t uncommitted_bytes; /* an interrupted write after the last commit; harmless */
} ShammyReport;

const char *shammy_version(void);
const char *shammy_last_error(void);
uint32_t    shammy_default_threads(void);

int32_t shammy_open(const char *path, ShammyReader **out);
void    shammy_close(ShammyReader *reader);
int32_t shammy_info(const ShammyReader *reader, ShammyInfo *out);
int32_t shammy_file_type(const ShammyReader *reader, uint32_t index, char *buf, size_t cap);

/* Up to k hits, best first, written to hits; the number written goes to count.
 * query_len may be less than dims (padded with zeros). file_types / doc_ids restrict the search;
 * pass NULL and 0 for no restriction. params may be NULL. */
int32_t shammy_search(const ShammyReader *reader, const float *query, size_t query_len, size_t k,
                      const char *const *file_types, size_t file_type_count,
                      const uint32_t *doc_ids, size_t doc_id_count,
                      const ShammyParams *params, ShammyHit *hits, size_t *count);

/* The stored vector of one chunk (dims values). found = 0 when it isn't stored. */
int32_t shammy_vector(const ShammyReader *reader, uint64_t id, float *out, size_t out_len, uint32_t *found);

/* Check every checksum. Returns SHAMMY_OK when the check ran; out->ok says whether the file is whole. */
int32_t shammy_verify(const char *path, ShammyReport *out);

/* ---------------------------------------------------------------------------------------------
 * The whole retrieval path over a RAG Workbench workspace or unpacked pack: vector search (Shammy
 * or sqlite-vec, whichever the knowledge base uses), FTS5 keyword search, rank fusion, duplicate
 * removal, optional reranking, neighbour expansion and citations. The Workbench and the retrieval
 * server both call this, so they return the same thing.
 *
 * The caller embeds the question and passes the vector. The request is a JSON object:
 *
 *   {"query": "how is the lid seal tested?",      required
 *    "k": 6,                                       results returned
 *    "file_type": "pdf",                           only this file type
 *    "expand": true, "neighbors": 1,               join prose hits with their neighbours
 *    "vec_k": 40, "fts_k": 40, "rrf_k": 60,        candidates per side and the fusion constant
 *    "max_result_tokens": 6000, "threads": 0,
 *    "markdown": true,                             also return the compact text form for an agent
 *    "rerank": {"enabled": true, "url": "http://127.0.0.1:8082", "model": "", "candidates": 20,
 *               "timeout_ms": 15000},              rerank through a /v1/rerank endpoint
 *    "procedures": true,                           with a spot file: a hit in a procedure brings it,
 *    "procedure_tokens": 1500,                       whole when this short, else the hit's step and
 *    "procedure_window": 1,                          up to this many steps each side that fit
 *    "documents": ["D02", "WI-2210"]}              only these documents (number or part of the path)
 *
 * The response is a JSON object: "results" (tag, chunk_id, kind, score, text, source{...},
 * rerank_score when reranked), "tables" (table cards that ranked), "candidates", "ranks",
 * "timings_ms", "rerank" ({"applied": true|false, "error": ...}) and "markdown" when asked for.
 * If the reranker can't be reached the search still succeeds in fused order and says so there.
 *
 * Structure (from 0.4.1): when the folder holds a structure.spot that matches its rag.db, results
 * also carry source.document / source.revision, "procedure" ({id, kind, title, label, steps, from,
 * to, whole}) for a hit inside a procedure, and "related" (references to other steps and
 * documents). Without one (any older pack), results are exactly as before.
 */
typedef struct ShammyRetriever ShammyRetriever;

int32_t shammy_rag_open(const char *folder, ShammyRetriever **out);  /* the folder holding rag.db */
void    shammy_rag_close(ShammyRetriever *retriever);
int32_t shammy_rag_info(const ShammyRetriever *retriever, char **out_json);
int32_t shammy_rag_search(const ShammyRetriever *retriever, const char *request_json,
                          const float *vector, size_t vector_len, char **out_json);
void    shammy_string_free(char *s);  /* for every char* this library returns through an out pointer */

#ifdef __cplusplus
}
#endif
#endif
