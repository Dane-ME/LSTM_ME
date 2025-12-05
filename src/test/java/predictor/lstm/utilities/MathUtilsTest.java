package predictor.lstm.utilities;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.assertEquals;

public class MathUtilsTest {

    private static final double TOLERANCE = 1e-6; // Sai số cho phép khi so sánh số thực

    @Test
    void testSigmoid() {
        // Test với giá trị 0
        assertEquals(0.5, MathUtils.sigmoid(0), TOLERANCE);

        // Test với giá trị dương lớn
        assertEquals(1.0, MathUtils.sigmoid(100), TOLERANCE);

        // Test với giá trị âm lớn
        assertEquals(0.0, MathUtils.sigmoid(-100), TOLERANCE);

        // Test với một giá trị cụ thể
        assertEquals(0.73105857863, MathUtils.sigmoid(1), TOLERANCE);
    }

    @Test
    void testSigmoidDerivative() {
        // The input to the derivative function should be 'x', not 'sigmoid(x)'.
        double x = 1.0;
        double expectedDerivative = MathUtils.sigmoid(x) * (1 - MathUtils.sigmoid(x));
        assertEquals(expectedDerivative, MathUtils.sigmoidDerivative(x), TOLERANCE);

        double x0 = 0.0;
        double expectedDerivativeAt0 = MathUtils.sigmoid(x0) * (1 - MathUtils.sigmoid(x0));
        assertEquals(expectedDerivativeAt0, MathUtils.sigmoidDerivative(x0), TOLERANCE);
    }

    @Test
    void testTanh() {
        // Test với giá trị 0
        assertEquals(0.0, MathUtils.tanh(0), TOLERANCE);

        // Test với giá trị dương lớn
        assertEquals(1.0, MathUtils.tanh(100), TOLERANCE);

        // Test với giá trị âm lớn
        assertEquals(-1.0, MathUtils.tanh(-100), TOLERANCE);

        // Test với một giá trị cụ thể
        assertEquals(0.76159415595, MathUtils.tanh(1), TOLERANCE);
    }

    @Test
    void testTanhDerivative() {
        // The input to the derivative function should be 'x', not 'tanh(x)'.
        double x = 1.0;
        double tanh_x = MathUtils.tanh(x);
        double expectedDerivative = 1 - (tanh_x * tanh_x);
        assertEquals(expectedDerivative, MathUtils.tanhDerivative(x), TOLERANCE);

        double x0 = 0.0;
        double tanh_x0 = MathUtils.tanh(x0);
        double expectedDerivativeAt0 = 1 - (tanh_x0 * tanh_x0);
        assertEquals(expectedDerivativeAt0, MathUtils.tanhDerivative(x0), TOLERANCE);
    }
}
