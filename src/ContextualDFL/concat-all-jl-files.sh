#!/bin/sh
find . -type f -name '*.jl' | while IFS= read -r file; do echo "// BEGIN OF $file"; cat "$file"; echo "// END OF $file" ; done
