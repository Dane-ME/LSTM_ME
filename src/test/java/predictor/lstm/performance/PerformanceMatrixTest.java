package predictor.lstm.performance;

import org.junit.jupiter.api.Test;
import java.util.ArrayList;
import java.util.Arrays;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

public class PerformanceMatrixTest {

    private static final double TOLERANCE = 1e-5;

    @Test
    void meanAbsoluteErrorTest() {
        ArrayList<Double> target = new ArrayList<>(Arrays.asList(1.0, 2.0, 3.0));
        ArrayList<Double> predicted = new ArrayList<>(Arrays.asList(2.0, 2.5, 2.8));
        assertEquals(0.56666, PerformanceMatrix.meanAbsoluteError(target, predicted), TOLERANCE);
    }

    @Test
    void meanAbsoluteErrorWithExceptionTest() {
        ArrayList<Double> target = new ArrayList<>(Arrays.asList(1.0, 2.0, 3.0));
        ArrayList<Double> predicted = new ArrayList<>(Arrays.asList(2.0, 2.5));
        assertThrows(IllegalArgumentException.class, () -> PerformanceMatrix.meanAbsoluteError(target, predicted));
    }

    @Test
    void rmsErrorTest() {
        ArrayList<Double> target = new ArrayList<>(Arrays.asList(1.0, 2.0, 3.0));
        ArrayList<Double> predicted = new ArrayList<>(Arrays.asList(2.0, 2.5, 2.8));
        assertEquals(0.65574, PerformanceMatrix.rmsError(target, predicted), TOLERANCE);
    }
    
    @Test
    void rmsErrorArrayTest() {
        double[] target = {1.0, 2.0, 3.0};
        double[] predicted = {2.0, 2.5, 2.8};
        assertEquals(0.65574, PerformanceMatrix.rmsError(target, predicted), TOLERANCE);
    }

    @Test
    void rmsErrorWithExceptionTest() {
        ArrayList<Double> target = new ArrayList<>(Arrays.asList(1.0, 2.0, 3.0));
        ArrayList<Double> predicted = new ArrayList<>(Arrays.asList(2.0, 2.5));
        assertThrows(IllegalArgumentException.class, () -> PerformanceMatrix.rmsError(target, predicted));
    }

    @Test
    void meanSquaredErrorTest() {
        ArrayList<Double> target = new ArrayList<>(Arrays.asList(1.0, 2.0, 3.0));
        ArrayList<Double> predicted = new ArrayList<>(Arrays.asList(2.0, 2.5, 2.8));
        assertEquals(0.43, PerformanceMatrix.meanSquaredError(target, predicted), TOLERANCE);
    }

    @Test
    void meanSquaredErrorExceptionTest() {
        ArrayList<Double> target = new ArrayList<>(Arrays.asList(1.0, 2.0, 3.0));
        ArrayList<Double> predicted = new ArrayList<>(Arrays.asList(2.0, 2.5));
        assertThrows(IllegalArgumentException.class, () -> PerformanceMatrix.meanSquaredError(target, predicted));
    }

    @Test
    void accuracyTest() {
        ArrayList<Double> target = new ArrayList<>(Arrays.asList(1.0, 2.0, 3.0));
        // New data: 1.09 is within 10% of 1.0. 2.8 is within 10% of 3.0. 2.15 is NOT within 10% of 2.0.
        ArrayList<Double> predicted = new ArrayList<>(Arrays.asList(1.09, 2.15, 2.8)); 
        double allowedPercentage = 0.1;
        assertEquals(2.0 / 3.0, PerformanceMatrix.accuracy(target, predicted, allowedPercentage), TOLERANCE);
    }
    
    @Test
    void accuracyArrayTest() {
        double[] target = {1.0, 2.0, 3.0};
        // New data: 1.09 is within 10% of 1.0. 2.8 is within 10% of 3.0. 2.15 is NOT within 10% of 2.0.
        double[] predicted = {1.09, 2.15, 2.8};
        double allowedPercentage = 0.1;
        assertEquals(2.0 / 3.0, PerformanceMatrix.accuracy(target, predicted, allowedPercentage), TOLERANCE);
    }

    @Test
    void accuracyWithEmptyListTest() {
        ArrayList<Double> target = new ArrayList<>();
        ArrayList<Double> predicted = new ArrayList<>();
        double allowedPercentage = 0.1;
        assertEquals(0.0, PerformanceMatrix.accuracy(target, predicted, TOLERANCE));
    }
}
