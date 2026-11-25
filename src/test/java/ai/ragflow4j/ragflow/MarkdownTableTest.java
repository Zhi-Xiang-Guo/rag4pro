// Copyright 2026 The InfiniFlow Authors. All Rights Reserved.
// Licensed under the Apache License, Version 2.0. See http://www.apache.org/licenses/LICENSE-2.0
// Distributed on an AS IS BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND.
package ai.ragflow4j.ragflow;

import static org.junit.jupiter.api.Assertions.*;

import java.util.*;

import org.junit.jupiter.api.Test;

/**
 * Fixtures derived from Go table_html_test.go and markdown_parser_test.go.
 */
class MarkdownTableTest {
    private String table(String markdown) {
        var result = new MarkdownParser().parse(markdown);
        assertTrue(result.warnings().isEmpty());
        return result.units().stream()
                .filter(u -> "table".equals(u.get("doc_type_kwd")))
                .findFirst()
                .orElseThrow()
                .get("text")
                .toString();
    }

    @Test
    void gfmTableMaterializesTheGoDeterministicMarkup() {
        assertEquals(
                "<table>\n"
                        + "<tr><th>Name</th><th>Age</th></tr>\n"
                        + "<tr><td>Alice</td><td>30</td></tr>\n"
                        + "</table>\n",
                table("| Name | Age |\n| --- | --- |\n| Alice | 30 |\n"));
    }

    @Test
    void raggedRowsPadHeaderBeforeGfmCanDiscardValues() {
        assertEquals(
                "<table>\n"
                        + "<tr><th>Name</th><th>Age</th><th></th><th></th></tr>\n"
                        + "<tr><td>Alice</td><td>30</td><td>Tail</td><td>Last</td></tr>\n"
                        + "</table>\n",
                table("| Name | Age |\n| --- | --- |\n| Alice | 30 | Tail | Last |\n"));
    }

    @Test
    void codeFencesRemainCodeEvenWithRaggedPipeRows() {
        var result =
                new MarkdownParser().parse("```\n| A | B |\n| --- | --- |\n| x | y | extra |\n```\n");
        assertEquals(1, result.units().size());
        assertEquals("code", result.units().getFirst().get("layout"));
        assertEquals(
                "| A | B |\n| --- | --- |\n| x | y | extra |", result.units().getFirst().get("text"));
    }

    @Test
    void rawTableIsSingleItemBetweenProseAndHasDefaultHeader() {
        var result =
                new MarkdownParser()
                        .parse("Before.\n\n<table><tr><td>X</td><td>Y</td></tr></table>\n\nAfter.\n");
        assertEquals(
                List.of("text", "table", "text"),
                result.units().stream().map(u -> u.get("doc_type_kwd")).toList());
        assertEquals(
                "<table>\n<tr><th>X</th><th>Y</th></tr>\n</table>\n", result.units().get(1).get("text"));
    }

    @Test
    void footerComesLastCaptionAndCellEscapingArePreserved() {
        assertEquals(
                "<table><caption>Stock &amp; price</caption>\n"
                        + "<tr><th>Item</th><th>Qty</th></tr>\n"
                        + "<tr><td>pen</td><td>1</td></tr>\n"
                        + "<tr><td>ink</td><td>2</td></tr>\n"
                        + "<tr><td>Total</td><td>3</td></tr>\n"
                        + "</table>\n",
                table(
                        "<table><caption>Stock &amp;"
                                + " price</caption><thead><tr><th>Item</th><th>Qty</th></tr></thead><tfoot><tr><th>Total</th><th>3</th></tr></tfoot><tbody><tr><td>pen</td><td>1</td></tr><tr><td>ink</td><td>2</td></tr></tbody></table>"));
    }

    @Test
    void mergedGridKeepsSpanGeometryAndNestedCellText() {
        assertEquals(
                "<table>\n"
                        + "<tr><th colspan=\"2\" rowspan=\"2\">A</th><th>B</th></tr>\n"
                        + "<tr><td>C\n"
                        + "Dinside</td></tr>\n"
                        + "</table>\n",
                table(
                        "<table><tr><th colspan='2'"
                                + " rowspan='2'>A</th><th>B</th></tr><tr><td>C<br>D<table><tr><td>inside</td></tr></table><script>drop</script></td></tr></table>"));
    }

    @Test
    void rawNonTableHtmlStaysVerbatimAsGoMarkdownRequires() {
        var result =
                new MarkdownParser()
                        .parse(
                                "Before.\n\n"
                                        + "<div class=\"note\">just a div block</div>\n\n"
                                        + "<style>.a{color:red}</style>\n\n"
                                        + "After.\n");
        assertTrue(
                result.units().stream()
                        .anyMatch(u -> u.get("text").equals("<div class=\"note\">just a div block</div>")));
        assertTrue(
                result.units().stream()
                        .anyMatch(u -> u.get("text").equals("<style>.a{color:red}</style>")));
    }

    @Test
    void htmlWalkerPreservesBlockBoundariesWhitespaceAndTableOrder() {
        var result =
                HtmlTextParser.parse(
                        "Hello"
                                + " <b>world</b>!<div>Name:<div>Alice</div><table><tr><td>one</td></tr></table>After<br>line</div><pre>"
                                + "  code\n"
                                + " next</pre>",
                        new LinkedHashSet<>());
        assertEquals(
                List.of(
                        "Hello world!",
                        "Name:\nAlice",
                        "<table>\n<tr><th>one</th></tr>\n</table>\n",
                        "After\nline",
                        "  code\n next"),
                result.stream().map(u -> u.get("text")).toList());
    }

    @Test
    void captionBrAndHtmlEscapesUseCellTextRatherThanMarkup() {
        assertEquals(
                "<table>\n"
                        + "<tr><th>A &amp; B</th><th>&#34;quote&#34; &#39;single&#39;</th></tr>\n"
                        + "<tr><td>line1\n"
                        + "line2</td><td>&lt;x&gt;</td></tr>\n"
                        + "</table>\n",
                table(
                        "<table><tr><th>A &amp; B</th><th>\"quote\""
                                + " 'single'</th></tr><tr><td>line1<br>line2</td><td>&lt;x&gt;</td></tr></table>"));
    }
}
