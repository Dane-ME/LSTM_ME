package predictor.lstm.preprocessing;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertEquals; // Added this import

public class DifferencingTest {

    @Test
    void testFirstOrderDifferencing() {
        double[] data = {10, 12, 15, 11, 13};
        double[] expected = {-2.0, -3.0, 4.0, -2.0};
        assertArrayEquals(expected, Differencing.firstOrderDifferencing(data), 1e-9);
    }
    
    @Test
    void testFirstOrderDifferencingWithTwoElements() {
        double[] data = {100, 90};
        double[] expected = {10.0};
        assertArrayEquals(expected, Differencing.firstOrderDifferencing(data), 1e-9);
    }

    @Test
    void testFirstOrderDifferencingNotEnoughData() {
        double[] data = {10};
        assertThrows(IllegalArgumentException.class, () -> Differencing.firstOrderDifferencing(data));
    }
    
    @Test
    void testFirstOrderAccumulating() {
        double[] data = {-2.0, -3.0, 4.0, -2.0};
        double init = 10.0;
        double[] expected = {8.0, 5.0, 9.0, 7.0};
        assertArrayEquals(expected, Differencing.firstOrderAccumulating(data, init), 1e-9);
    }

    @Test
    void testFirstOrderAccumulatingNotEnoughData() {
        double[] data = {};
        assertThrows(IllegalArgumentException.class, () -> Differencing.firstOrderAccumulating(data, 10.0));
    }
    
    @Test
    void testSingleValueFirstOrderAccumulating() {
        double data = 5.5;
        double init = 10.0;
        assertEquals(15.5, Differencing.firstOrderAccumulating(data, init), 1e-9);
    }
}
