package predictor.lstm.preprocessingpipeline;

import java.util.Arrays;

public class ZeroToNanPipe implements Stage<Object, Object> {

    private static final double ZERO_THRESHOLD = 1e-6; // A small threshold to handle floating point inaccuracies

    @Override
    public Object execute(Object input) {
        if (input instanceof double[] in) {
            return Arrays.stream(in)
                    .map(value -> (Math.abs(value) < ZERO_THRESHOLD) ? Double.NaN : value)
                    .toArray();
        } else {
            // If the input is not a double array, pass it through without modification.
            // Or throw an exception if the pipeline should strictly handle double arrays.
            // For now, we'll just pass it through.
            System.err.println("Warning: ZeroToNanPipe received an input that is not a double array. Type: " + input.getClass().getName());
            return input;
        }
    }
}
