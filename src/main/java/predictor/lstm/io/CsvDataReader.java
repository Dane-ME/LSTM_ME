package predictor.lstm.io;

import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;
import org.apache.commons.csv.CSVRecord;
import predictor.lstm.data.TimeSeriesData;

import java.io.FileReader;
import java.io.IOException;
import java.io.Reader;
import java.time.LocalDateTime;
import java.time.OffsetDateTime;
import java.time.ZoneId;
import java.time.format.DateTimeFormatter;
import java.time.format.DateTimeFormatterBuilder;
import java.time.format.DateTimeParseException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static java.time.temporal.ChronoField.*;

public class CsvDataReader {

    private static final List<DateTimeFormatter> FORMATTERS = Arrays.asList(
            // Định dạng LINH HOẠT (cho phép 1 hoặc 2 chữ số cho giờ/phút)
            createFlexibleFormatter("M/d/yyyy H:m"),       // 10/10/2025 9:00
            createFlexibleFormatter("d/M/yyyy H:m"),       // 10/10/2025 9:00
            createFlexibleFormatter("yyyy-M-d H:m"),       // 2025-10-10 9:00
            createFlexibleFormatter("d-M-yyyy H:m"),       // 10-10-2025 9:00

            // Định dạng CHUẨN (yêu cầu 2 chữ số)
            DateTimeFormatter.ofPattern("MM/dd/yyyy HH:mm"),
            DateTimeFormatter.ofPattern("dd/MM/yyyy HH:mm"),
            DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm"),
            DateTimeFormatter.ofPattern("dd-MM-yyyy HH:mm"),
            DateTimeFormatter.ofPattern("yyyy/MM/dd HH:mm"),

            // Định dạng có giây
            createFlexibleFormatter("M/d/yyyy H:m:s"),
            createFlexibleFormatter("d/M/yyyy H:m:s"),
            DateTimeFormatter.ofPattern("MM/dd/yyyy HH:mm:ss"),
            DateTimeFormatter.ofPattern("dd/MM/yyyy HH:mm:ss"),
            DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss"),
            DateTimeFormatter.ofPattern("dd-MM-yyyy HH:mm:ss"),
            DateTimeFormatter.ofPattern("yyyy/MM/dd HH:mm:ss"),
            DateTimeFormatter.ofPattern("yyyy-MM-dd'T'HH:mm:ss"),
            DateTimeFormatter.ISO_LOCAL_DATE_TIME
    );

    private static final ZoneId DEFAULT_ZONE = ZoneId.of("Asia/Ho_Chi_Minh");
    private static final boolean DEBUG = false; // Tắt debug sau khi fix

    /**
     * Tạo formatter linh hoạt cho phép 1 hoặc 2 chữ số
     */
    private static DateTimeFormatter createFlexibleFormatter(String pattern) {
        return new DateTimeFormatterBuilder()
                .appendPattern(pattern)
                .parseDefaulting(SECOND_OF_MINUTE, 0)
                .toFormatter();
    }

    public TimeSeriesData read(String filePath) throws IOException {
        ArrayList<OffsetDateTime> dates = new ArrayList<>();
        ArrayList<Double> values = new ArrayList<>();

        int skippedRows = 0;
        int processedRows = 0;
        DateTimeFormatter detectedFormatter = null;

        System.out.println("=== Reading CSV: " + filePath + " ===");

        try (Reader reader = new FileReader(filePath);
             CSVParser csvParser = new CSVParser(reader, CSVFormat.DEFAULT
                     .withFirstRecordAsHeader()
                     .withTrim()
                     .withIgnoreEmptyLines())) {

            if (csvParser.getHeaderNames() != null && !csvParser.getHeaderNames().isEmpty()) {
                System.out.println("Headers: " + csvParser.getHeaderNames());
            }

            for (CSVRecord csvRecord : csvParser) {
                processedRows++;

                if (DEBUG && processedRows <= 5) {
                    System.out.println("\n--- Row " + csvRecord.getRecordNumber() + " ---");
                    for (int i = 0; i < csvRecord.size(); i++) {
                        System.out.println("Col " + i + ": [" + csvRecord.get(i) + "]");
                    }
                }

                if (csvRecord.size() < 2) {
                    skippedRows++;
                    continue;
                }

                try {
                    String timestampStr = csvRecord.get(0);
                    String valueStr = csvRecord.get(1);

                    if (timestampStr == null || timestampStr.trim().isEmpty()) {
                        skippedRows++;
                        continue;
                    }

                    timestampStr = timestampStr.trim();

                    LocalDateTime ldt;
                    if (detectedFormatter == null) {
                        ldt = parseWithAutoDetect(timestampStr);
                        if (ldt != null) {
                            detectedFormatter = findWorkingFormatter(timestampStr);
                            System.out.println("✓ Detected format for: [" + timestampStr + "]");
                            System.out.println("✓ Parsed as: " + ldt);
                        } else {
                            if (skippedRows < 3) {
                                System.err.println("✗ Cannot parse: [" + timestampStr + "]");
                            }
                            skippedRows++;
                            continue;
                        }
                    } else {
                        ldt = LocalDateTime.parse(timestampStr, detectedFormatter);
                    }

                    OffsetDateTime odt = ldt.atZone(DEFAULT_ZONE).toOffsetDateTime();

                    Double value = null;
                    if (valueStr != null && !valueStr.trim().isEmpty()) {
                        value = Double.parseDouble(valueStr.trim());

                        if (value.isNaN() || value.isInfinite()) {
                            skippedRows++;
                            continue;
                        }
                    }

                    dates.add(odt);
                    values.add(value);

                } catch (DateTimeParseException e) {
                    if (skippedRows < 3) {
                        System.err.println("Row " + csvRecord.getRecordNumber() + ": Date error [" + csvRecord.get(0) + "]");
                    }
                    skippedRows++;
                } catch (NumberFormatException e) {
                    if (skippedRows < 3) {
                        System.err.println("Row " + csvRecord.getRecordNumber() + ": Number error [" + csvRecord.get(1) + "]");
                    }
                    skippedRows++;
                } catch (Exception e) {
                    if (skippedRows < 3) {
                        System.err.println("Row " + csvRecord.getRecordNumber() + ": Error - " + e.getMessage());
                    }
                    skippedRows++;
                }

                // Progress indicator cho file lớn
                if (processedRows % 2000 == 0) {
                    System.out.println("Processing... " + processedRows + " rows, loaded: " + dates.size());
                }
            }
        }

        System.out.println("\n=== Summary ===");
        System.out.println("Total rows: " + processedRows);
        System.out.println("Loaded: " + dates.size());
        System.out.println("Skipped: " + skippedRows);

        if (dates.isEmpty()) {
            throw new IOException("No valid data in file: " + filePath);
        }

        System.out.println("✓ Successfully loaded " + dates.size() + " records");
        return new TimeSeriesData(dates, values);
    }

    private LocalDateTime parseWithAutoDetect(String timestampStr) {
        for (DateTimeFormatter formatter : FORMATTERS) {
            try {
                return LocalDateTime.parse(timestampStr, formatter);
            } catch (DateTimeParseException e) {
                // Try next formatter
            }
        }
        return null;
    }

    private DateTimeFormatter findWorkingFormatter(String timestampStr) {
        for (DateTimeFormatter formatter : FORMATTERS) {
            try {
                LocalDateTime.parse(timestampStr, formatter);
                return formatter;
            } catch (DateTimeParseException e) {
                // Try next formatter
            }
        }
        return FORMATTERS.get(0);
    }
}