# Native lexical analyzer provenance

The `upstream` directory is an unmodified copy of RAGFlow's `internal/binding/cpp` at commit `e4bd0801d5169c3dc0847ce0ea06fc6c3e516105`. Its Apache-2.0 license is included. Embedded OpenCC, RE2, Snowball and Darts retain their source license headers.

The build compiles the same analyzer and C ABI used by Go. `free_string.cpp` adds allocator-matched release of returned strings. Java serializes access to an owned analyzer, preserving native mutable language state and destroying the handle on close.

Run `scripts/native-build.sh` on macOS or Linux. Linux packages are `g++ libpcre2-dev libsimde-dev`. macOS uses Command Line Tools and Homebrew `pcre2 simde`. The resulting shared library is loaded through JNA. Set `RAGFLOW_NATIVE_LIBRARY` to override its absolute path and `RAGFLOW_DICT_PATH` to override the dictionary directory. Defaults resolve under `native/` from the application working directory.

The packaged lexical resources come from `https://github.com/infiniflow/resource`, revision `0937399b60f1949267388548e33ea0d5c0cc25f7`, the resource source used by upstream's `docker/launch_backend_service.sh`. Only analyzer-required files are packaged. `scripts/native-resources.sh` reproduces resource acquisition at that exact revision. The original prebuilt `rag/huqie.trie` is included, so analyzer loading needs no dictionary-directory writes.

The included `resources/synonym.json` is copied byte-for-byte from RAGFlow `internal/rag/res/synonym.json` at the source revision above. `WordNet.java` and `Synonym.java` port `internal/service/nlp/wordnet.go` and `synonym.go`. WordNet data files retain their own bundled license. The Go oracle fixture tests complete synset fields for six words, including inflected forms.
