package predictor.lstm.validator.forgetgate;

import java.time.OffsetDateTime;
import java.util.ArrayList;

import predictor.lstm.common.DataStatistics;
import predictor.lstm.common.HyperParameters;
import predictor.lstm.common.forgetgate.LstmPredictorWithForgetGate;

public class ValidationTrendModelWithForgetGate {

    public void validateTrend(ArrayList<Double> data, //
                              ArrayList<OffsetDateTime> date, //
                              ArrayList<ArrayList<ArrayList<ArrayList<Double>>>> allModels,
                              HyperParameters hyperParameters) {

        var rmsErrorTrend = new ArrayList<Double>();

        for (int i = 0; i < allModels.size(); i++) {
            hyperParameters.updatModelTrend(allModels.get(i));
            var predicted = LstmPredictorWithForgetGate.predictTrend(data, date,
                    date.get(date.size() - 1).atZoneSameInstant(date.get(0).getOffset()), hyperParameters, allModels.get(i));
            int predictionSize = predicted.size();
            double[] actualSlice = data.subList(0, predictionSize).stream().mapToDouble(Double::doubleValue).toArray();
            double[] predictedArray = predicted.stream().mapToDouble(Double::doubleValue).toArray();
            double error = DataStatistics.computeRms(actualSlice, predictedArray);
            rmsErrorTrend.add(error);
            hyperParameters.setRmsErrorTrend(error);
        }
        hyperParameters.setAllModelErrorTrend(rmsErrorTrend);
    }
}
