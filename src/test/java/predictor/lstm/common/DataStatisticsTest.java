package predictor.lstm.common;

import org.junit.jupiter.api.Test;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

public class DataStatisticsTest {

    private static final double TOLERANCE = 1e-5;

    @Test
    void testGetMeanFromCollection() {
        ArrayList<Double> data = new ArrayList<>(Arrays.asList(1.0, 2.0, 3.0, 4.0, 5.0));
        assertEquals(3.0, DataStatistics.getMean(data), TOLERANCE);
    }

    @Test
    void testGetMeanFromEmptyCollection() {
        ArrayList<Double> emptyData = new ArrayList<>();
        assertEquals(0.0, DataStatistics.getMean(emptyData), TOLERANCE);
    }

    @Test
    void testGetMeanFrom2DArray() {
        double[][] data = {{1.0, 2.0}, {3.0, 4.0, 5.0}};
        double[] expected = {1.5, 4.0};
        double[] actual = DataStatistics.getMean(data);
        assertEquals(expected.length, actual.length);
        for (int i = 0; i < expected.length; i++) {
            assertEquals(expected[i], actual[i], TOLERANCE);
        }
    }

    @Test
    void testGetMeanFrom1DArray() {
        double[] data = {1.0, 2.0, 3.0, 4.0, 5.0};
        assertEquals(3.0, DataStatistics.getMean(data), TOLERANCE);
    }

    @Test
    void testGetStandardDeviationFromCollection() {
        ArrayList<Double> data = new ArrayList<>(Arrays.asList(1.0, 2.0, 3.0, 4.0, 5.0));
        assertEquals(1.41421356, DataStatistics.getStandardDeviation(data), TOLERANCE);
    }
    
    @Test
    void testGetStandardDeviationFrom1DArray() {
        double[] data = {1.0, 2.0, 3.0, 4.0, 5.0};
        assertEquals(1.41421356, DataStatistics.getStandardDeviation(data), TOLERANCE);
    }

    @Test
    void testGetStandardDeviationWithZeroVariance() {
        ArrayList<Double> data = new ArrayList<>(Arrays.asList(5.0, 5.0, 5.0, 5.0));
        // Should return a very small number instead of 0 to avoid division by zero errors later
        assertEquals(0.000000000000001, DataStatistics.getStandardDeviation(data), TOLERANCE);
    }
    
    @Test
    void testComputeRms() {
        double[] original = {1.0, 2.0, 3.0, 4.0, 5.0};
        double[] computed = {1.1, 2.2, 3.1, 4.2, 5.1};
        double expectedRms = 0.14832396974191304;
        assertEquals(expectedRms, DataStatistics.computeRms(original, computed), TOLERANCE);
    }

    @Test
    void testComputeRmsWithDifferentLengths() {
        double[] original = {1.0, 2.0, 3.0};
        double[] computed = {1.1, 2.2};
        assertThrows(IllegalArgumentException.class, () -> {
            DataStatistics.computeRms(original, computed);
        });
    }
}
