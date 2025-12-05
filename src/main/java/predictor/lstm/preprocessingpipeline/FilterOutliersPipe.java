package predictor.lstm.preprocessingpipeline;

import predictor.lstm.preprocessing.FilterOutliers;

public class FilterOutliersPipe implements Stage<Object, Object> {

    @Override
    public Object execute(Object input) {
        return input instanceof double[] in //
                ? FilterOutliers.filterOutlier(in) //
                : new IllegalArgumentException("Input must be an instance of double[]");
    }
}
