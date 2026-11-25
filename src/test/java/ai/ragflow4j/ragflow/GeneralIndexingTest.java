// Copyright 2026 The InfiniFlow Authors. All Rights Reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and limitations.
package ai.ragflow4j.ragflow;

import static org.junit.jupiter.api.Assertions.*;

import java.util.*;

import org.junit.jupiter.api.Test;

class GeneralIndexingTest {
    private final Cl100k tokens = new Cl100k();

    private List<String> texts(List<Map<String, Object>> chunks) {
        return chunks.stream().map(c -> c.get("text").toString()).toList();
    }

    @Test
    void defaultTextRetainsDelimitersAndNormalizedNewlines() {
        String text = "alpha!\r\n\r\nbeta?\rgamma。";
        assertEquals(
                List.of("alpha!\n\nbeta?\ngamma。"),
                texts(new GeneralChunker(tokens).chunkText(text, GeneralChunker.Config.defaults())));
    }

    @Test
    void overCapAllowsBoundaryOverflowAndRetainsOversizedAtoms() {
        var chunker = new GeneralChunker(tokens);
        var config = new GeneralChunker.Config(tokens.count("alpha!"), List.of("!"), List.of(), 0);
        assertEquals(
                List.of("alpha!beta!", "gamma!"), texts(chunker.chunkText("alpha!beta!gamma!", config)));
        String longAtom = "oversized atom stays whole ".repeat(50);
        assertEquals(List.of(longAtom), texts(chunker.chunkText(longAtom, config)));
    }

    @Test
    void customDelimiterDropsMarkersAndBypassesMerge() {
        var config = new GeneralChunker.Config(512, List.of("`END`"), List.of(), 0);
        assertEquals(
                List.of("alpha", "beta", "gamma"),
                texts(new GeneralChunker(tokens).chunkText("alphaENDbetaENDgamma", config)));
        assertEquals(
                List.of("alphaendbeta"),
                texts(new GeneralChunker(tokens).chunkText("alphaendbeta", config)));
    }

    @Test
    void childrenKeepDelimiterAndCarryFullParent() {
        var config = new GeneralChunker.Config(512, List.of("!"), List.of(";"), 0);
        var result = new GeneralChunker(tokens).chunkText("first;second;third", config);
        assertEquals(List.of("first;", "second;", "third"), texts(result));
        result.forEach(c -> assertEquals("first;second;third", c.get("mom")));
    }

    @Test
    void markdownShortHeadingForcesNextUnitSourceFixture() {
        var units = List.of(GeneralChunker.unit("Title", "heading"), GeneralChunker.unit("body", ""));
        var config = new GeneralChunker.Config(1, List.of("\n"), List.of(), 0);
        assertEquals(
                List.of("Title\nbody"), texts(new GeneralChunker(tokens).chunkMarkdown(units, config)));
    }

    @Test
    void markdownOverlapSourceFixture() {
        var units =
                List.of("alpha beta", "gamma delta", "epsilon zeta", "eta theta").stream()
                        .map(t -> GeneralChunker.unit(t, ""))
                        .toList();
        int target = tokens.count("alpha beta") + tokens.count("gamma delta");
        var config = new GeneralChunker.Config(target, List.of("\n"), List.of(), 50);
        assertEquals(
                List.of("alpha beta\ngamma delta", "gamma delta\nepsilon zeta", "epsilon zeta\neta theta"),
                texts(new GeneralChunker(tokens).chunkMarkdown(units, config)));
    }

    @Test
    void markdownBlockquoteAndListsMirrorGoFixtures() {
        var parser = new MarkdownParser();
        assertEquals(
                List.of("First quote\nSecond quote"),
                texts(parser.parse("> First quote\n>\n> Second quote\n").units()));
        assertEquals(
                List.of("Intro\nitem one\nitem two"),
                texts(parser.parse("> Intro\n>\n> - item one\n> - item two\n").units()));
        assertEquals(List.of("10. a", "11. b"), texts(parser.parse("10. a\n11. b\n").units()));
        assertEquals(
                List.of("See also", "* China at the Asian Games", "* China at the Paralympics"),
                texts(
                        parser
                                .parse("See also\n\n* China at the Asian Games\n* China at the Paralympics\n")
                                .units()));
    }

    @Test
    void markdownHeadingAndCodeKeepSemanticLayouts() {
        var result =
                new MarkdownParser()
                        .parse("\ufeff# Hello\n\nThis is a paragraph.\n\n```go\nfunc main() {}\n```\n");
        assertEquals(
                List.of("# Hello", "This is a paragraph.", "func main() {}"), texts(result.units()));
        assertEquals("heading", result.units().getFirst().get("layout"));
        assertEquals("code", result.units().getLast().get("layout"));
        assertTrue(result.warnings().isEmpty());
    }

    @Test
    void specializedMarkdownBranchesReportTheirLimits() {
        assertFalse(
                new MarkdownParser().parse("![alt](https://example.com/a.png)").warnings().isEmpty());
        var table = new MarkdownParser().parse("| A | B |\n| --- | --- |\n| 1 | 2 |\n");
        assertEquals("table", table.units().getFirst().get("doc_type_kwd"));
        assertTrue(table.units().getFirst().get("text").toString().contains("<td>2</td>"));
        assertTrue(table.warnings().isEmpty());
    }

    private static final IndexingPipeline.LexicalTokenizer LEXICAL =
            new IndexingPipeline.LexicalTokenizer() {
                public String tokenize(String text, String language) {
                    return text.toLowerCase(Locale.ROOT);
                }

                public String fineGrainedTokenize(String text, String language) {
                    return "fine:" + text;
                }
            };

    @Test
    void tokenizerFieldsAndRawTitleBlendMatchSource() throws Exception {
        List<List<String>> calls = new ArrayList<>();
        var embedder =
                new IndexingPipeline.Embedder() {
                    public List<double[]> embed(List<String> texts) {
                        calls.add(List.copyOf(texts));
                        return texts.stream()
                                .map(t -> calls.size() == 1 ? new double[]{10, 20} : new double[]{2, 4})
                                .toList();
                    }
                };
        var unit = GeneralChunker.unit("content", "");
        unit.put("questions", "one\ntwo");
        unit.put("keywords", "alpha,,beta，gamma");
        var result =
                new IndexingPipeline(LEXICAL, tokens, embedder)
                        .process(" Raw.PDF ", List.of(unit), IndexingPipeline.Config.defaults("English"));
        assertEquals(List.of(" Raw.PDF "), calls.getFirst());
        assertEquals(" raw.pdf ", result.getFirst().get("title_tks"));
        assertEquals("content", result.getFirst().get("content_ltks"));
        assertEquals(List.of("one", "two"), result.getFirst().get("question_kwd"));
        assertEquals(List.of("alpha", "", "beta，gamma"), result.getFirst().get("important_kwd"));
        assertArrayEquals(new double[]{2.8, 5.6}, (double[]) result.getFirst().get("q_2_vec"), 1e-12);
        assertEquals(0, result.getFirst().get("chunk_order_int"));
        assertFalse(unit.containsKey("content_ltks"));
    }

    @Test
    void emptyTitleSkipsBlendAndEmbeddingLimitsMatchSource() throws Exception {
        List<List<String>> calls = new ArrayList<>();
        var embedder =
                new IndexingPipeline.Embedder() {
                    public List<double[]> embed(List<String> texts) {
                        calls.add(texts);
                        return List.of(new double[]{2, 4});
                    }
                };
        var result =
                new IndexingPipeline(LEXICAL, tokens, embedder)
                        .process(
                                "  ",
                                List.of(GeneralChunker.unit("content", "")),
                                IndexingPipeline.Config.defaults("English"));
        assertEquals(1, calls.size());
        assertArrayEquals(new double[]{2, 4}, (double[]) result.getFirst().get("q_2_vec"));
        assertEquals(2007, IndexingPipeline.embeddingTokenLimit(0));
        assertEquals(480, IndexingPipeline.embeddingTokenLimit(512));
        assertEquals(5, IndexingPipeline.embeddingTokenLimit(10));
    }

    @Test
    void declaredTitlesPreserveNonAsciiAndFilterLabelsStopwordsNumbers() {
        assertEquals(
                List.of("pragyan", "ojha", "三国演义", "icc", "cup"),
                IndexingPipeline.declaredTitleTokens(
                        List.of(
                                GeneralChunker.unit(
                                        "title: The Pragyan Ojha 2024\nname: 《三国演义》 Wikipedia\nfullname: ICC Cup",
                                        ""))));
    }

    @Test
    void embeddingInputConcatenatesContextAndFieldsWithoutInventedSeparators() throws Exception {
        List<String> inputs = new ArrayList<>();
        var embedder =
                new IndexingPipeline.Embedder() {
                    public List<double[]> embed(List<String> texts) {
                        inputs.addAll(texts);
                        return texts.stream().map(t -> new double[]{1}).toList();
                    }
                };
        var unit = GeneralChunker.unit("body", "");
        unit.put("context_above", "above ");
        unit.put("context_below", " below");
        unit.put("questions", "question");
        var config =
                new IndexingPipeline.Config("English", true, true, 0.1, List.of("text", "questions"));
        var result = new IndexingPipeline(LEXICAL, tokens, embedder).process("", List.of(unit), config);
        assertEquals(List.of("above body belowquestion"), inputs);
        assertEquals("above body below", result.getFirst().get("content_ltks"));
    }
}
