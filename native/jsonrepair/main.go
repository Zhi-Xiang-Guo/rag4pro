// This executable preserves RAGFlow's exact kaptinlin/jsonrepair dependency.
// The caller owns Markdown/think-tag cleanup and JSON decoding, as in
// internal/service/metadata_filter.go GenMetaFilter.
package main

import (
	"fmt"
	"io"
	"os"

	"github.com/kaptinlin/jsonrepair"
)

func main() {
	input, err := io.ReadAll(os.Stdin)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	repaired, err := jsonrepair.Repair(string(input))
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	if _, err = io.WriteString(os.Stdout, repaired); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
