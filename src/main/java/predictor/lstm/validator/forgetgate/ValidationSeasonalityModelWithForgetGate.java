package predictor.lstm.validator.forgetgate;

import java.time.OffsetDateTime;
import java.util.ArrayList;

import predictor.lstm.common.DataStatistics;
import predictor.lstm.common.HyperParameters;
import predictor.lstm.common.forgetgate.LstmPredictorWithForgetGate;

public class ValidationSeasonalityModelWithForgetGate {

    public void validateSeasonality(ArrayList<Double> data, //
                                    ArrayList<OffsetDateTime> date, //
                                    ArrayList<ArrayList<ArrayList<ArrayList<Double>>>> allModels,
                                    HyperParameters hyperParameters) {

        var rmsErrorSeasonality = new ArrayList<Double>();

        for (int i = 0; i < allModels.size(); i++) {
            hyperParameters.updateModelSeasonality(allModels.get(i));
            var predicted = LstmPredictorWithForgetGate.predictSeasonality(data, date, hyperParameters, allModels.get(i));
            int predictionSize = predicted.size();
            double[] actualSlice = data.subList(0, predictionSize).stream().mapToDouble(Double::doubleValue).toArray();
            double[] predictedArray = predicted.stream().mapToDouble(Double::doubleValue).toArray();
            double error = DataStatistics.computeRms(actualSlice, predictedArray);
            rmsErrorSeasonality.add(error);
            hyperParameters.setRmsErrorSeasonality(error);
        }
        hyperParameters.setAllModelErrorSeason(rmsErrorSeasonality);
    }
}
