package predictor.lstm.train.forgetgate;

import static predictor.lstm.preprocessing.DataModification.constantScaling;
import static predictor.lstm.preprocessing.DataModification.removeNegatives;

import java.io.IOException;
import java.time.OffsetDateTime;
import java.util.ArrayList;
import java.util.Objects;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import common.types.ChannelAddress;
import predictor.lstm.common.ReadAndSaveModels;
import predictor.lstm.data.TimeSeriesData;
import predictor.lstm.io.CsvDataReader;
import predictor.lstm.preprocessing.TimeIndexRegularizer;

public class LstmTrainWithForgetGate implements Runnable {

    private final Logger log = LoggerFactory.getLogger(LstmTrainWithForgetGate.class);

    private final ChannelAddress channelAddress;
    private final long days;
    private final String filePath;

    public LstmTrainWithForgetGate(ChannelAddress channelAddress, long days, String filePath) {
        this.channelAddress = channelAddress;
        this.days = days;
        this.filePath = filePath;
    }

    @Override
    public void run() {
        var hyperParameters = ReadAndSaveModels.read(this.channelAddress.getChannelId());
        ArrayList<Double> trainingData;
        ArrayList<OffsetDateTime> trainingDate;
        ArrayList<Double> validationData;
        ArrayList<OffsetDateTime> validationDate;

        try {
            CsvDataReader reader = new CsvDataReader();
            TimeSeriesData rawData = reader.read(this.filePath);

            int interval = hyperParameters.getInterval();
            TimeSeriesData allData = TimeIndexRegularizer.regularize(rawData.dates(), rawData.values(), interval);

            ArrayList<OffsetDateTime> allDates = allData.dates();
            ArrayList<Double> allValues = allData.values();

            if (this.days > 0) {
                int requiredPoints = (int) (this.days * 24 * 60 / interval);
                if (allValues.size() > requiredPoints) {
                    int startIndex = allValues.size() - requiredPoints;
                    allValues = new ArrayList<>(allValues.subList(startIndex, allValues.size()));
                    allDates = new ArrayList<>(allDates.subList(startIndex, allDates.size()));
                }
            }

            int totalItems = allValues.size();
            int trainSize = (int) (totalItems * 0.66); // 66% train and 33% validation

            trainingData = new ArrayList<>(allValues.subList(0, trainSize));
            trainingDate = new ArrayList<>(allDates.subList(0, trainSize));

            validationData = new ArrayList<>(allValues.subList(trainSize, totalItems));
            validationDate = new ArrayList<>(allDates.subList(trainSize, totalItems));

        } catch (IOException e) {
            this.log.error("Failed to read or process CSV file: " + this.filePath, e);
            return;
        }

        if (this.cannotTrainConditions(trainingData)) {
            this.log.info("Cannot proceed with training: Data is all null or insufficient data.");
            return;
        }
        ReadAndSaveModels.adapt(hyperParameters, constantScaling(removeNegatives(validationData), 1), validationDate);
        TrainWithForgetGate trainer = new TrainWithForgetGate(
                constantScaling(removeNegatives(trainingData), 1),
                trainingDate,
                constantScaling(removeNegatives(validationData), 1),
                validationDate,
                hyperParameters);
        trainer.setEarlyStoppingEnabled(true);
        trainer.setEarlyStoppingPatience(5);
    }

    private boolean cannotTrainConditions(ArrayList<Double> array) {
        if (array.isEmpty()) {
            this.log.info("Array is empty");
            return true;
        }

        boolean allNulls = array.stream().allMatch(Objects::isNull);
        if (allNulls) {
            this.log.info("allNulls");
            return true;
        }

        var nonNanCount = array.stream().filter(d -> d != null && !Double.isNaN(d)).count();
        var validProportion = (double) nonNanCount / array.size();
        return validProportion <= 0.5;
    }
}
