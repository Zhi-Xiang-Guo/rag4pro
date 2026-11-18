package ai.ragflow4j.ragflow;

import org.apache.pdfbox.Loader;
import org.apache.pdfbox.pdmodel.PDPage;
import org.apache.pdfbox.pdmodel.encryption.InvalidPasswordException;
import org.apache.pdfbox.text.PDFTextStripper;
import org.apache.poi.xwpf.usermodel.XWPFDocument;
import org.apache.poi.xwpf.usermodel.XWPFParagraph;
import org.apache.poi.xwpf.usermodel.XWPFTable;

import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.io.StringWriter;
import java.nio.ByteBuffer;
import java.nio.charset.CharacterCodingException;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Set;
import java.util.stream.Collectors;

import static ai.ragflow4j.ragflow.DocumentSources.Unit;

/** The extracted order and unit numbers tie indexed text to the original file. */
public final class DocumentParser {
    public static final int MAX_BYTES = 16 * 1024 * 1024;
    private static final int MAX_TEXT_LENGTH = 2_000_000;
    private static final int MAX_PDF_PAGES = 1000;
    private static final Set<String> FORMATS = Set.of("txt", "md", "pdf", "docx");

    private DocumentParser() {}

    public static String format(String filename) {
        int dot = filename.lastIndexOf('.');
        String format = dot < 0 ? "" : filename.substring(dot + 1).toLowerCase(Locale.ROOT);
        if (!FORMATS.contains(format)) throw new IllegalArgumentException("支持 TXT、Markdown（.md）、PDF 和 DOCX");
        return format;
    }

    public static List<Unit> parse(String format, byte[] original) throws IOException {
        if (original.length == 0) throw new IllegalArgumentException("文件为空");
        if (original.length > MAX_BYTES) throw new IllegalArgumentException("文件超过 16 MiB");
        var extraction = new Extraction();
        switch (format) {
            case "txt", "md" -> extraction.add("text", 1, utf8(original));
            case "pdf" -> extractPdf(original, extraction);
            case "docx" -> extractWord(original, extraction);
            default -> throw new IllegalArgumentException("不支持的文档格式");
        }
        if (extraction.units.isEmpty()) {
            throw new IllegalArgumentException(format.equals("pdf")
                    ? "PDF 未提取到文字，扫描件需要 OCR" : "文档没有可索引的正文");
        }
        return List.copyOf(extraction.units);
    }

    private static String utf8(byte[] bytes) {
        try {
            String text = StandardCharsets.UTF_8.newDecoder().decode(ByteBuffer.wrap(bytes)).toString();
            return text.startsWith("\uFEFF") ? text.substring(1) : text;
        } catch (CharacterCodingException error) {
            throw new IllegalArgumentException("请使用 UTF-8 编码的文本文件", error);
        }
    }

    private static void extractPdf(byte[] bytes, Extraction extraction) throws IOException {
        try (var pdf = Loader.loadPDF(bytes)) {
            if (pdf.getNumberOfPages() > MAX_PDF_PAGES)
                throw new IllegalArgumentException("PDF 页数超过 1000 页");
            var text = new StringWriter();
            var stripper = new PDFTextStripper() {
                @Override
                protected void endPage(PDPage page) {
                    extraction.add("page", getCurrentPageNo(), text.toString());
                    text.getBuffer().setLength(0);
                }
            };
            stripper.setSortByPosition(true);
            stripper.writeText(pdf, text);
        } catch (InvalidPasswordException error) {
            throw new IllegalArgumentException("请上传已解密的 PDF", error);
        }
    }

    private static void extractWord(byte[] bytes, Extraction extraction) throws IOException {
        try (var word = new XWPFDocument(new ByteArrayInputStream(bytes))) {
            int paragraphs = 0, tables = 0;
            for (var element : word.getBodyElements()) {
                if (element instanceof XWPFParagraph paragraph) {
                    extraction.add("paragraph", ++paragraphs, paragraph.getText());
                } else if (element instanceof XWPFTable table) {
                    String text = table.getRows().stream().map(row -> row.getTableCells().stream()
                            .map(cell -> cell.getTextRecursively()).collect(Collectors.joining("\t")))
                            .collect(Collectors.joining("\n"));
                    extraction.add("table", ++tables, text);
                }
            }
        }
    }

    private static final class Extraction {
        private final List<Unit> units = new ArrayList<>();
        private int length;

        void add(String kind, int number, String text) {
            String normalized = GeneralChunker.normalize(text);
            if (normalized.isBlank()) return;
            length += normalized.length();
            if (length > MAX_TEXT_LENGTH) throw new IllegalArgumentException("文档抽取的正文超过 200 万字符");
            units.add(new Unit(kind, number, normalized));
        }
    }
}
