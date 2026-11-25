#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
source_dir="$root/native/upstream"
build_dir="$root/native/build"
mkdir -p "$build_dir"
compiler="${CXX:-c++}"
flags=(-O2 -fPIC -pthread -I"$source_dir")
link_flags=(-lpcre2-8)
case "$(uname -s)" in
  Darwin)
    prefix="$(brew --prefix)"
    flags+=(-I"$prefix/include")
    link_flags=(-L"$prefix/lib" -lpcre2-8)
    shared_flag=-dynamiclib
    extension=dylib
    ;;
  Linux) shared_flag=-shared; extension=so ;;
  *) echo "Unsupported native platform" >&2; exit 1 ;;
esac
sources=()
while IFS= read -r file; do
  case "$file" in
    *_test.cpp|*/main.cpp|*/rag_analyzer_c_api_debug.cpp) continue ;;
  esac
  sources+=("$file")
done < <(find "$source_dir" -type f \( -name '*.cpp' -o -name '*.cc' -o -name '*.c' \) | sort)
objects=()
for file in "${sources[@]}" "$root/native/free_string.cpp"; do
  relative="${file#"$root/native/"}"
  object="$build_dir/objects/$relative.o"
  mkdir -p "$(dirname "$object")"
  if [[ "$file" == *.c ]]; then
    "${CC:-cc}" -std=gnu11 "${flags[@]}" -c "$file" -o "$object"
  else
    "$compiler" -std=c++20 "${flags[@]}" -c "$file" -o "$object"
  fi
  objects+=("$object")
done
"$compiler" "$shared_flag" "${objects[@]}" "${link_flags[@]}" -o "$build_dir/libragflow_tokenizer.$extension"
printf '%s\n' "$build_dir/libragflow_tokenizer.$extension"
