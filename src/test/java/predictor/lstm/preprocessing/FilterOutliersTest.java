package predictor.lstm.preprocessing;

import java.util.ArrayList; // Added this import
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class FilterOutliersTest {

    @Test
    void testFilterOutlier() {
        double[] data = {10, 12, 12, 13, 12, 11, 14, 13, 15, 10, 1000, -1000};
        double[] filtered = FilterOutliers.filterOutlier(data);

        // In this case, 1000 and -1000 are clear outliers.
        // The test checks if they have been "squashed" by the tanh function.
        // The exact value after tanh isn't critical, just that it's no longer the extreme value.
        // tanh(1000) is ~1.0. tanh(-1000) is ~-1.0.
        
        assertEquals(12, filtered.length);
        assertTrue(filtered[10] < 1000 && filtered[10] > 0); // Check if 1000 was squashed
        assertTrue(filtered[11] > -1000 && filtered[11] < 0); // Check if -1000 was squashed
        assertEquals(10, filtered[0], 1e-9); // Check that a non-outlier is unchanged
    }

    @Test
    void testDetect() {
        double[] data = {10, 12, 12, 13, 12, 11, 14, 13, 15, 10, 1000}; // 1000 is an outlier
        ArrayList<Integer> outlierIndices = FilterOutliers.detect(data);
        assertEquals(1, outlierIndices.size());
        assertEquals(10, outlierIndices.get(0).intValue());
    }

    @Test
    void testDetectNoOutliers() {
        double[] data = {10, 12, 12, 13, 12, 11, 14, 13, 15, 10};
        ArrayList<Integer> outlierIndices = FilterOutliers.detect(data);
        assertTrue(outlierIndices.isEmpty());
    }
    
    @Test
    void testEmptyInput() {
        double[] emptyData = {};
        assertThrows(IllegalArgumentException.class, () -> FilterOutliers.filterOutlier(emptyData));
        assertThrows(IllegalArgumentException.class, () -> FilterOutliers.detect(emptyData));
    }
}
