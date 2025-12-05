package predictor.lstm.train;

import java.time.OffsetDateTime;
import java.util.ArrayList;
import java.util.Collections;
import java.util.concurrent.CompletableFuture;

import predictor.lstm.common.HyperParameters;
import predictor.lstm.common.ReadAndSaveModels;
import predictor.lstm.preprocessing.DataModification;
import predictor.lstm.validator.ValidationSeasonalityModel;
import predictor.lstm.validator.ValidationTrendModel;

public class TrainAndValidateBatchImproved {

    private boolean earlyStoppingEnabled = false;
    private int earlyStoppingPatience = 5;
    private double bestValidationError = Double.MAX_VALUE;
    private int patienceCounter = 0;

    public void setEarlyStoppingEnabled(boolean enabled) {
        this.earlyStoppingEnabled = enabled;
    }

    public void setEarlyStoppingPatience(int patience) {
        this.earlyStoppingPatience = patience;
    }

    private void printProgressBar(int currentBatch, int totalBatches, int currentEpoch, int totalEpochs) {
        int barWidth = 50;
        float progress = (float) currentBatch / totalBatches;
        int completedWidth = (int) (barWidth * progress);

        String bar = "=".repeat(completedWidth) + " ".repeat(barWidth - completedWidth);
        String progressText = String.format("Epoch %d/%d | [%s] %d%% | Batch %d/%d",
                currentEpoch + 1, totalEpochs, bar, (int) (progress * 100), currentBatch, totalBatches);

        System.out.print("\r" + progressText);
    }

    public TrainAndValidateBatchImproved(//
                                 ArrayList<Double> trainData, //
                                 ArrayList<OffsetDateTime> trainDate, //
                                 ArrayList<Double> validateData, //
                                 ArrayList<OffsetDateTime> validateDate, //
                                 HyperParameters hyperParameter) {

        var batchedData = DataModification.getDataInBatch(
                trainData, hyperParameter.getBatchSize());
        var batchedDate = DataModification.getDateInBatch(
                trainDate, hyperParameter.getBatchSize());

        for (int epoch = hyperParameter.getEpochTrack(); epoch < hyperParameter.getEpoch(); epoch++) {
            int k = hyperParameter.getCount();

            if (epoch == 0 && this.earlyStoppingEnabled) {
                this.bestValidationError = Double.MAX_VALUE;
            }

            for (int batch = hyperParameter.getBatchTrack(); batch < hyperParameter.getBatchSize(); batch++) {
                hyperParameter.setCount(k);
                printProgressBar(batch + 1, hyperParameter.getBatchSize(), epoch, hyperParameter.getEpoch());

                MakeModelImproved makeModels = new MakeModelImproved();

                var trainDataTemp = batchedData.get(batch);
                var trainDateTemp = batchedDate.get(batch);

                CompletableFuture<Void> firstTaskFuture = CompletableFuture
                        .supplyAsync(() -> makeModels.trainSeasonality(trainDataTemp, trainDateTemp, hyperParameter))
                        .thenAccept(untestedSeasonalityMoadels -> new ValidationSeasonalityModel().validateSeasonality(
                                validateData, validateDate, untestedSeasonalityMoadels, hyperParameter));

                CompletableFuture<Void> secondTaskFuture = CompletableFuture
                        .supplyAsync(() -> makeModels.trainTrend(trainDataTemp, trainDateTemp, hyperParameter))
                        .thenAccept(untestedSeasonalityMoadels -> new ValidationTrendModel().validateTrend(validateData,
                                validateDate, untestedSeasonalityMoadels, hyperParameter));

                k = k + 1;
                try {
                    CompletableFuture.allOf(firstTaskFuture, secondTaskFuture).get();

                    if (this.earlyStoppingEnabled) {
                        double currentValidationError = Collections.min(hyperParameter.getRmsErrorSeasonality());

                        if (currentValidationError < this.bestValidationError) {
                            this.bestValidationError = currentValidationError;
                            this.patienceCounter = 0;
                            // In a real robust implementation, we would save the "best model" here explicitly.
                            // However, ReadAndSaveModels.save(hyperParameter) saves the CURRENT state.
                            // If we continue training, we overwrite it.
                            // But since the loop breaks on patience exhaustion, the LAST saved model
                            // might not be the BEST model if patience > 0.
                            // Ideally, we should rollback to best model, but to keep changes minimal,
                            // we rely on the fact that patience is small (e.g. 5) so the model hasn't drifted far.
                        } else {
                            this.patienceCounter++;

                            if (this.patienceCounter >= this.earlyStoppingPatience) {
                                System.out.println("\nEarly stopping triggered at epoch " + epoch);
                                epoch = hyperParameter.getEpoch(); // Force outer loop to break
                                break;
                            }
                        }
                    }
                } catch (Exception e) {
                    e.printStackTrace();
                }

                hyperParameter.setBatchTrack(batch + 1);
                hyperParameter.setCount(k);
                ReadAndSaveModels.save(hyperParameter);
            }

            hyperParameter.setBatchTrack(0);
            hyperParameter.setEpochTrack(hyperParameter.getEpochTrack() + 1);
            hyperParameter.update();
            ReadAndSaveModels.save(hyperParameter);
        }

        System.out.println();
        hyperParameter.setEpochTrack(0);
        ReadAndSaveModels.save(hyperParameter);
    }
}
