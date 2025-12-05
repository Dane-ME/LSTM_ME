package predictor.lstm.train.forgetgate;

import static predictor.lstm.utilities.UtilityConversion.to1DArray;

import java.time.OffsetDateTime;
import java.util.ArrayList;

import predictor.lstm.common.DynamicItterationValue;
import predictor.lstm.common.HyperParameters;
import predictor.lstm.preprocessingpipeline.PreprocessingPipeImpl;
import predictor.lstm.util.forgetgate.EngineWithForgetGate;
import predictor.lstm.util.forgetgate.EngineWithForgetGate.EngineBuilder;

public class MakeModelWithForgetGate {

    public synchronized ArrayList<ArrayList<ArrayList<ArrayList<Double>>>> trainTrend(ArrayList<Double> data,
                                                                                      ArrayList<OffsetDateTime> date, HyperParameters hyperParameters) {
        var weightMatrix = new ArrayList<ArrayList<ArrayList<ArrayList<Double>>>>();
        var weightTrend = new ArrayList<ArrayList<Double>>();
        PreprocessingPipeImpl preProcessing = new PreprocessingPipeImpl(hyperParameters);
        preProcessing.setData(to1DArray(data));
        preProcessing.setDates(date);

        var modifiedData = (double[][]) preProcessing//
                .replaceZerosWithNaNs()//
                .interpolate()//
                .movingAverage()//
                .scale()//
                .filterOutliers()//
                .modifyForTrendPrediction()//
                .execute();

        for (int i = 0; i < modifiedData.length; i++) {
            weightTrend = (hyperParameters.getCount() == 0) //
                    ? generateInitialWeightMatrix(hyperParameters.getWindowSizeTrend(), hyperParameters)//
                    : hyperParameters.getlastModelTrend().get(i);

            preProcessing.setData(modifiedData[i]);

            var preProcessed = (double[][][]) preProcessing//
                    .groupToStiffedWindow()//
                    .normalize()//
                    .shuffle()//
                    .execute();

            var model = new EngineBuilder() //
                    .setInputMatrix(preProcessed[0])//
                    .setTargetVector(preProcessed[1][0]) //
                    .build();
            model.fit(hyperParameters.getGdIterration(), weightTrend, hyperParameters);
            weightMatrix.add(model.getWeights());
        }
        return weightMatrix;
    }

    public synchronized ArrayList<ArrayList<ArrayList<ArrayList<Double>>>> trainSeasonality(ArrayList<Double> data,
                                                                                            ArrayList<OffsetDateTime> date, HyperParameters hyperParameters) {
        var weightMatrix = new ArrayList<ArrayList<ArrayList<ArrayList<Double>>>>();
        var weightSeasonality = new ArrayList<ArrayList<Double>>();
        int windowsSize = hyperParameters.getWindowSizeSeasonality();

        var preprocessing = new PreprocessingPipeImpl(hyperParameters);
        preprocessing.setData(to1DArray(data));//
        preprocessing.setDates(date);//

        var dataGroupedByMinute = (double[][][]) preprocessing//
                .replaceZerosWithNaNs()//
                .interpolate()//
                .movingAverage()//
                .scale()//
                .filterOutliers()//
                .groupByHoursAndMinutes()//
                .execute();
        int k = 0;

        for (int i = 0; i < dataGroupedByMinute.length; i++) {
            for (int j = 0; j < dataGroupedByMinute[i].length; j++) {

                hyperParameters.setGdIterration(DynamicItterationValue
                        .setIteration(hyperParameters.getAllModelErrorSeason(), k, hyperParameters));

                if (hyperParameters.getCount() == 0) {
                    weightSeasonality = generateInitialWeightMatrix(windowsSize, hyperParameters);
                } else {
                    weightSeasonality = hyperParameters.getlastModelSeasonality().get(k);
                }

                preprocessing.setData(dataGroupedByMinute[i][j]);

                var preProcessedSeason = (double[][][]) preprocessing//
                        .groupToWIndowSeasonality() //
                        .normalize() //
                        .shuffle() //
                        .execute();

                var model = new EngineBuilder()//
                        .setInputMatrix(preProcessedSeason[0]) //
                        .setTargetVector(preProcessedSeason[1][0]) //
                        .build();

                model.fit(hyperParameters.getGdIterration(), weightSeasonality, hyperParameters);
                weightMatrix.add(model.getWeights());
                k = k + 1;
            }
        }
        return weightMatrix;
    }

    public static ArrayList<ArrayList<Double>> generateInitialWeightMatrix(int windowSize,
                                                                           HyperParameters hyperParameters) {
        var initialWeight = new ArrayList<ArrayList<Double>>();
        // Extended parameter types
        var parameterTypes = new String[] { "wi", "wo", "wz", "ri", "ro", "rz", "yt", "ct", "wf", "rf" };

        for (var type : parameterTypes) {
            var temp = new ArrayList<Double>();
            for (int i = 1; i <= windowSize; i++) {
                var value = switch (type) {
                    case "wi" -> hyperParameters.getWiInit();
                    case "wo" -> hyperParameters.getWoInit();
                    case "wz" -> hyperParameters.getWzInit();
                    case "ri" -> hyperParameters.getRiInit();
                    case "ro" -> hyperParameters.getRoInit();
                    case "rz" -> hyperParameters.getRzInit();
                    case "yt" -> hyperParameters.getYtInit();
                    case "ct" -> hyperParameters.getCtInit();
                    case "wf" -> hyperParameters.getWfInit();
                    case "rf" -> hyperParameters.getRfInit();
                    default -> throw new IllegalArgumentException("Invalid parameter type");
                };
                temp.add(value);
            }
            initialWeight.add(temp);
        }
        return initialWeight;
    }
}
