#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
revision=0937399b60f1949267388548e33ea0d5c0cc25f7
base="https://raw.githubusercontent.com/infiniflow/resource/$revision"
files=(LICENSE wordnet/LICENSE rag/huqie.txt rag/huqie.trie rag/pos-id.def opencc/zht2zhs.ini
       opencc/trad_to_simp_phrases.ocd opencc/trad_to_simp_characters.ocd)
for name in noun verb adj adv; do
  files+=("wordnet/index.$name" "wordnet/$name.exc" "wordnet/data.$name")
done
for file in "${files[@]}"; do
  destination="$root/native/resources/$file"
  mkdir -p "$(dirname "$destination")"
  curl --fail --location --retry 3 --connect-timeout 20 --max-time 180 \
    "$base/$file" -o "$destination.download"
  mv "$destination.download" "$destination"
done
printf '%s\n' "$revision" > "$root/native/resources/REVISION"
