package predictor.lstm.train;

import static predictor.lstm.utilities.UtilityConversion.to1DArray;

import java.time.OffsetDateTime;
import java.util.ArrayList;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import predictor.lstm.common.DynamicItterationValue;
import predictor.lstm.common.HyperParameters;
import predictor.lstm.preprocessingpipeline.PreprocessingPipeImpl;
import predictor.lstm.util.improved.EngineImproved;
import predictor.lstm.util.improved.EngineImproved.EngineBuilder;

public class MakeModelImproved {
    private final Logger log = LoggerFactory.getLogger(MakeModelImproved.class);


    public static final String SEASONALITY = "seasonality";
    public static final String TREND = "trend";

    /**
     * Trains the trend model using the specified data, timestamps, and
     * hyperparameters.
     */
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
                    ? MakeModel.generateInitialWeightMatrix(hyperParameters.getWindowSizeTrend(), hyperParameters)//
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

    /**
     * Trains the seasonality model using the specified data, timestamps, and
     * hyperparameters.
     */

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
                    weightSeasonality = MakeModel.generateInitialWeightMatrix(windowsSize, hyperParameters);

                } else {
                    weightSeasonality = hyperParameters.getlastModelSeasonality().get(k);
                }

                preprocessing.setData(dataGroupedByMinute[i][j]);

                var preProcessedSeason = (double[][][]) preprocessing//
                        // .differencing()//
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
}
