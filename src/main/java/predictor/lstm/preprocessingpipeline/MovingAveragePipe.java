package predictor.lstm.preprocessingpipeline;


import predictor.lstm.preprocessing.MovingAverage;

public class MovingAveragePipe implements Stage<Object, Object> {

    @Override
    public Object execute(Object input) {
        return input instanceof double[] in //
                ? MovingAverage.movingAverage(in) //
                : new IllegalArgumentException("Input must be an instance of double[]");
    }
}