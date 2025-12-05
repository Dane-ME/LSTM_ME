package predictor.lstm.preprocessing;

import org.junit.jupiter.api.Test;
import java.time.OffsetDateTime;
import java.util.ArrayList;
import java.util.Arrays;
import static org.junit.jupiter.api.Assertions.*;

public class DataModificationTest {

    @Test
    void testScale() {
        ArrayList<Double> testData = new ArrayList<>(Arrays.asList(10.0, 20.0, 30.0));
        ArrayList<Double> scaledData = DataModification.scale(testData, 0.0, 40.0);
        // Formula: 0.2 + ((val - min) / (max - min)) * (0.8 - 0.2)
        // 10.0 -> 0.2 + ((10-0)/(40-0)) * 0.6 = 0.2 + 0.25 * 0.6 = 0.2 + 0.15 = 0.35
        // 20.0 -> 0.2 + ((20-0)/(40-0)) * 0.6 = 0.2 + 0.5 * 0.6 = 0.2 + 0.3 = 0.5
        // 30.0 -> 0.2 + ((30-0)/(40-0)) * 0.6 = 0.2 + 0.75 * 0.6 = 0.2 + 0.45 = 0.65
        assertEquals(0.35, scaledData.get(0), 1e-6);
        assertEquals(0.5, scaledData.get(1), 1e-6);
        assertEquals(0.65, scaledData.get(2), 1e-6);
    }

    @Test
    void testScaleBack() {
        double scaledValue = 0.5;
        double minOriginal = 0.0;
        double maxOriginal = 40.0;
        double originalValue = DataModification.scaleBack(scaledValue, minOriginal, maxOriginal);
        assertEquals(20.0, originalValue, 1e-6);
    }

    @Test
    void testGetDataInBatch() {
        ArrayList<Double> data = new ArrayList<>(Arrays.asList(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0));
        int numberOfGroups = 3;
        ArrayList<ArrayList<Double>> result = DataModification.getDataInBatch(data, numberOfGroups);
        assertEquals(numberOfGroups, result.size());
        assertEquals(Arrays.asList(1.0, 2.0, 3.0, 4.0), result.get(0));
        assertEquals(Arrays.asList(5.0, 6.0, 7.0), result.get(1));
        assertEquals(Arrays.asList(8.0, 9.0, 10.0), result.get(2));
    }

    @Test
    void testGetDateInBatch() {
        ArrayList<OffsetDateTime> dateList = new ArrayList<>();
        OffsetDateTime startingDate = OffsetDateTime.parse("2023-01-01T00:00:00Z");
        for (int i = 0; i < 10; i++) {
            dateList.add(startingDate.plusMinutes(i * 15));
        }
        int numberOfGroups = 3;
        ArrayList<ArrayList<OffsetDateTime>> result = DataModification.getDateInBatch(dateList, numberOfGroups);
        assertEquals(numberOfGroups, result.size());
        assertEquals(4, result.get(0).size());
        assertEquals(3, result.get(1).size());
        assertEquals(3, result.get(2).size());
        assertEquals(startingDate.plusMinutes(5 * 15), result.get(1).get(1));
    }

    @Test
    void testRemoveNegatives() {
        ArrayList<Double> inputList = new ArrayList<>(Arrays.asList(5.0, -3.0, 2.0, -7.5, null));
        ArrayList<Double> expectedList = new ArrayList<>(Arrays.asList(5.0, 0.0, 2.0, 0.0, Double.NaN));
        ArrayList<Double> resultList = DataModification.removeNegatives(inputList);

        for (int i = 0; i < expectedList.size(); i++) {
            if (Double.isNaN(expectedList.get(i))) {
                assertTrue(Double.isNaN(resultList.get(i)));
            } else {
                assertEquals(expectedList.get(i), resultList.get(i), 1e-9);
            }
        }
    }

    @Test
    void testConstantScaling() {
        ArrayList<Double> inputData = new ArrayList<>(Arrays.asList(1.0, 2.0, 3.0));
        double scalingFactor = 2.0;
        ArrayList<Double> expectedOutput = new ArrayList<>(Arrays.asList(2.0, 4.0, 6.0));
        ArrayList<Double> actualOutput = DataModification.constantScaling(inputData, scalingFactor);
        assertEquals(expectedOutput, actualOutput);
    }
     @Test
    void elementWiseMultiplicationTest() {
        double[] featureA = { 1.0, 2.0, 3.0 };
        double[] featureB = { 4.0, 5.0, 6.0 };
        double[] expected = { 4.0, 10.0, 18.0 };
        assertArrayEquals(expected, DataModification.elementWiseMultiplication(featureA, featureB), 1e-9);
    }

    @Test
    void elementWiseDivisionTest() {
        ArrayList<Double> featureA = new ArrayList<>(Arrays.asList(10.0, 20.0, 30.0));
        ArrayList<Double> featureB = new ArrayList<>(Arrays.asList(2.0, 5.0, 4.0));
        ArrayList<Double> expected = new ArrayList<>(Arrays.asList(5.0, 4.0, 7.5));
        assertEquals(expected, DataModification.elementWiseDiv(featureA, featureB));
    }

    @Test
    void elementWiseDivisionByZeroTest() {
        ArrayList<Double> featureA = new ArrayList<>(Arrays.asList(10.0, 20.0));
        ArrayList<Double> featureB = new ArrayList<>(Arrays.asList(0.0, 5.0));
        // If divisor is 0, it should return the original value from featureA
        ArrayList<Double> expected = new ArrayList<>(Arrays.asList(10.0, 4.0));
        assertEquals(expected, DataModification.elementWiseDiv(featureA, featureB));
    }
}
