package ai.ragflow4j.ragflow;

import org.apache.pdfbox.pdmodel.PDDocument;
import org.apache.pdfbox.pdmodel.PDPage;
import org.apache.pdfbox.pdmodel.PDPageContentStream;
import org.apache.pdfbox.pdmodel.font.PDType1Font;
import org.apache.pdfbox.pdmodel.font.Standard14Fonts;
import org.apache.poi.xwpf.usermodel.XWPFDocument;
import org.junit.jupiter.api.Test;

import java.io.ByteArrayOutputStream;
import java.nio.charset.StandardCharsets;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

class DocumentParserTest {
    @Test void identicalPdfTextKeepsPageNumbersAndBlankPagesDoNotShiftThem() throws Exception {
        var output = new ByteArrayOutputStream();
        try (var doc = new PDDocument()) {
            for (int number = 1; number <= 3; number++) {
                var page = new PDPage();
                doc.addPage(page);
                if (number == 2) continue;
                try (var pageText = new PDPageContentStream(doc, page)) {
                    pageText.beginText();
                    pageText.setFont(new PDType1Font(Standard14Fonts.FontName.HELVETICA), 12);
                    pageText.newLineAtOffset(25, 685);
                    pageText.showText("Repeated on two pages.");
                    pageText.endText();
                }
            }
            doc.save(output);
        }
        var units = DocumentParser.parse("pdf", output.toByteArray());
        assertEquals(List.of(1, 3), units.stream().map(DocumentSources.Unit::number).toList());
        assertEquals(units.getFirst().text(), units.getLast().text());
        assertTrue(units.getFirst().text().contains("Repeated on two pages."));
    }

    @Test void docxPreservesBodyOrderAndTableCells() throws Exception {
        var output = new ByteArrayOutputStream();
        try (var word = new XWPFDocument()) {
            word.createParagraph().createRun().setText("开头");
            var table = word.createTable(1, 2);
            table.getRow(0).getCell(0).setText("列一");
            table.getRow(0).getCell(1).setText("列二");
            word.createParagraph().createRun().setText("后文");
            word.write(output);
        }
        var units = DocumentParser.parse("docx", output.toByteArray());
        assertEquals(List.of(
                new DocumentSources.Unit("paragraph", 1, "开头"),
                new DocumentSources.Unit("table", 1, "列一\t列二"),
                new DocumentSources.Unit("paragraph", 2, "后文")), units);
    }

    @Test void textIsDecodedWithStableLineBreaksAndAnEmptyPdfRequestsOcr() throws Exception {
        assertEquals("# text\n\nline", DocumentParser.parse("md",
                "\uFEFF# text\r\n\r\nline".getBytes(StandardCharsets.UTF_8)).getFirst().text());
        var output = new ByteArrayOutputStream();
        try (var doc = new PDDocument()) {
            doc.addPage(new PDPage());
            doc.save(output);
        }
        var error = assertThrows(IllegalArgumentException.class, () -> DocumentParser.parse("pdf", output.toByteArray()));
        assertTrue(error.getMessage().contains("OCR"));
    }
}
