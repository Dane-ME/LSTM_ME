package predictor.lstm.common.forgetgate;

import static predictor.lstm.utilities.UtilityConversion.to1DArray;
import static predictor.lstm.utilities.UtilityConversion.to1DArrayList;
import static predictor.lstm.utilities.UtilityConversion.to2DArrayList;
import static predictor.lstm.utilities.UtilityConversion.to2DList;

import java.time.OffsetDateTime;
import java.time.ZonedDateTime;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import predictor.lstm.common.DataStatistics;
import predictor.lstm.common.HyperParameters;
import predictor.lstm.common.LstmPredictor;
import predictor.lstm.data.TimeSeriesData;
import predictor.lstm.preprocessing.TimeIndexRegularizer;
import predictor.lstm.preprocessingpipeline.PreprocessingPipeImpl;
import predictor.lstm.utilities.MathUtils;

public class LstmPredictorWithForgetGate {

    public static ArrayList<Double> predictSeasonality(ArrayList<Double> data, ArrayList<OffsetDateTime> date,
                                                       HyperParameters hyperParameters) {
        return predictSeasonality(data, date, hyperParameters, hyperParameters.getBestModelSeasonality());
    }

    public static ArrayList<Double> predictSeasonality(ArrayList<Double> data, ArrayList<OffsetDateTime> date,
                                                       HyperParameters hyperParameters,
                                                       ArrayList<ArrayList<ArrayList<Double>>> model) {
        int interval = hyperParameters.getInterval();
        TimeSeriesData regularizedInput = TimeIndexRegularizer.regularize(date, data, interval);

        var preprocessing = new PreprocessingPipeImpl(hyperParameters);
        preprocessing.setData(to1DArray(regularizedInput.values())).setDates(regularizedInput.dates());
        var resized = to2DList((double[][][]) preprocessing.interpolate()//
                .scale()//
                .movingAverage()
                .filterOutliers() //
                .groupByHoursAndMinutes()//
                .execute());
        preprocessing.setData(resized);
        var normalized = (double[][]) preprocessing//
                .normalize()//
                .execute();
        var predicted = predictPre(to2DArrayList(normalized), model, hyperParameters);

        // Flatten resized 2D array to 1D array to calculate global statistics
        double[] flattenedResized = Arrays.stream(resized).flatMapToDouble(Arrays::stream).toArray();

        double mean = (hyperParameters.getMean() != 0) ? hyperParameters.getMean() : DataStatistics.getMean(flattenedResized);
        double stdDev = (hyperParameters.getStandardDeviation() != 1) ? hyperParameters.getStandardDeviation() : DataStatistics.getStandardDeviation(flattenedResized);

        preprocessing.setData(to1DArray(predicted))//
                .setMean(mean)
                .setStandardDeviation(stdDev);
        var seasonalityPrediction = (double[]) preprocessing.reverseNormalize()//
                .reverseScale()//
                .execute();
        return to1DArrayList(seasonalityPrediction);
    }


    public static ArrayList<Double> predictTrend(ArrayList<Double> data, ArrayList<OffsetDateTime> date,
                                                 ZonedDateTime until, HyperParameters hyperParameters) {
        return predictTrend(data, date, until, hyperParameters, hyperParameters.getBestModelTrend());
    }

    public static ArrayList<Double> predictTrend(ArrayList<Double> data, ArrayList<OffsetDateTime> date,
                                                 ZonedDateTime until, HyperParameters hyperParameters,
                                                 ArrayList<ArrayList<ArrayList<Double>>> model) {
        int interval = hyperParameters.getInterval();
        TimeSeriesData regularizedInput = TimeIndexRegularizer.regularize(date, data, interval);

        var preprocessing = new PreprocessingPipeImpl(hyperParameters);
        preprocessing.setData(to1DArray(regularizedInput.values())).setDates(regularizedInput.dates());

        var scaled = (double[]) preprocessing//
                .interpolate()//
                .scale()//
                .movingAverage()
                .filterOutliers()
                .execute();

        // normalize
        var trendPrediction = new double[hyperParameters.getTrendPoint()];

        double mean = (hyperParameters.getMean() != 0) ? hyperParameters.getMean() : DataStatistics.getMean(scaled);
        double stdDev = (hyperParameters.getStandardDeviation() != 1) ? hyperParameters.getStandardDeviation() : DataStatistics.getStandardDeviation(scaled);

        preprocessing.setData(scaled);
        var normData = to1DArrayList((double[]) preprocessing//
                .normalize()//
                .execute());

        var predictionFor = until.plusMinutes(hyperParameters.getInterval());
        var val = model;

        for (int i = 0; i < hyperParameters.getTrendPoint(); i++) {
            var temp = predictionFor.plusMinutes(i * hyperParameters.getInterval());

            var modlelindex = (int) LstmPredictor.decodeDateToColumnIndex(temp, hyperParameters);
            double predTemp = predict(//
                    normData, //
                    val.get(modlelindex).get(0), val.get(modlelindex).get(1), //
                    val.get(modlelindex).get(2), val.get(modlelindex).get(3), //
                    val.get(modlelindex).get(4), val.get(modlelindex).get(5), //
                    (val.get(modlelindex).size() > 8) ? val.get(modlelindex).get(8) : null, // Wf
                    (val.get(modlelindex).size() > 9) ? val.get(modlelindex).get(9) : null, // Rf
                    val.get(modlelindex).get(7), val.get(modlelindex).get(6), //
                    hyperParameters);
            normData.add(predTemp);
            normData.remove(0);
            trendPrediction[i] = (predTemp);
        }

        preprocessing.setData(trendPrediction).setMean(mean).setStandardDeviation(stdDev);

        return to1DArrayList((double[]) preprocessing//
                .reverseNormalize()//
                .reverseScale()//
                .execute());
    }

    public static ArrayList<Double> predictPre(ArrayList<ArrayList<Double>> inputData,
                                               ArrayList<ArrayList<ArrayList<Double>>> val, HyperParameters hyperParameters) {
        var result = new ArrayList<Double>();
        for (var i = 0; i < inputData.size(); i++) {

            var wi = val.get(i).get(0);
            var wo = val.get(i).get(1);
            var wz = val.get(i).get(2);
            var rI = val.get(i).get(3);
            var rO = val.get(i).get(4);
            var rZ = val.get(i).get(5);
            var ct = val.get(i).get(7);
            var yt = val.get(i).get(6);

            ArrayList<Double> wf = (val.get(i).size() > 8) ? val.get(i).get(8) : null;
            ArrayList<Double> rf = (val.get(i).size() > 9) ? val.get(i).get(9) : null;
            result.add(predict(inputData.get(i), wi, wo, wz, rI, rO, rZ, wf, rf, ct, yt, hyperParameters));
        }
        return result;
    }

    public static double predict(ArrayList<Double> inputData, ArrayList<Double> wi, ArrayList<Double> wo,
                                 ArrayList<Double> wz, ArrayList<Double> rI, ArrayList<Double> rO, ArrayList<Double> rZ,
                                 ArrayList<Double> Wf, ArrayList<Double> Rf,
                                 ArrayList<Double> cta, ArrayList<Double> yta, HyperParameters hyperParameters) {
        var ct = hyperParameters.getCtInit();
        var yt = hyperParameters.getYtInit();

        int windowSize = wi.size();
        List<Double> standData = inputData.subList(Math.max(0, inputData.size() - windowSize), inputData.size());

        for (var i = 0; i < standData.size(); i++) {
            var ctMinusOne = ct;
            var yTMinusOne = yt;
            var xt = standData.get(i);

            var it = MathUtils.sigmoid(wi.get(i) * xt + rI.get(i) * yTMinusOne);
            var ot = MathUtils.sigmoid(wo.get(i) * xt + rO.get(i) * yTMinusOne);
            var zt = MathUtils.tanh(wz.get(i) * xt + rZ.get(i) * yTMinusOne);

            double ft = 1.0;
            if (Wf != null && Rf != null) {
                ft = MathUtils.sigmoid(Wf.get(i) * xt + Rf.get(i) * yTMinusOne);
            }

            ct = ft * ctMinusOne + it * zt;
            yt = ot * MathUtils.tanh(ct);
        }
        return yt;
    }

    // Helper to reuse getIndex from original class
    public static Integer getIndex(Integer hour, Integer minute, HyperParameters hyperParameters) {
        return LstmPredictor.getIndex(hour, minute, hyperParameters);
    }

    // Helper to reuse getArranged from original class
    public static ArrayList<Double> getArranged(int splitIndex, ArrayList<Double> singleArray) {
        return LstmPredictor.getArranged(splitIndex, singleArray);
    }
}
