package predictor.lstm.common;

import java.io.Serializable;
import java.time.OffsetDateTime;
import java.util.ArrayList;
import java.util.Collections;

public class HyperParameters implements Serializable {

    private OffsetDateTime lastTrainedDate;

    public OffsetDateTime getLastTrainedDate() {
        return this.lastTrainedDate;
    }

    private static final long serialVersionUID = 1L;

    private final int maxItterFactor = 10;

    private double learningRateUpperLimit = 0.02;

    private double learnignRateLowerLimit = 0.00005;

    private double dataSplitTrain = 0.7;

    private double dataSplitValidate = 1 - this.dataSplitTrain;

    private double wiInit = 0.2;
    private double woInit = 0.2;
    private double wzInit = 0.2;
    private double riInit = 0.2;
    private double roInit = 0.2;
    private double rzInit = 0.2;
    private double ytInit = 0.2;
    private double ctInit = 0.2;
    private double wfInit = 1.0;
    private double rfInit = 1.0;

    private int interval = 30;

    private int batchSize = 1;

    private int batchTrack = 0;

    private int epoch = 50;

    private int epochTrack = 0;

    private int trendPoints = 7;

    private int windowSizeSeasonality = 14;

    private int windowSizeTrend = 7;

    private int gdIterration = 55;

    private int count = 0;

    private double targetError = 0;

    private double scalingMin = 0;

    private double scalingMax = 1000;

    private ArrayList<ArrayList<ArrayList<ArrayList<Double>>>> modelTrend = new ArrayList<ArrayList<ArrayList<ArrayList<Double>>>>();

    private ArrayList<ArrayList<ArrayList<ArrayList<Double>>>> modelSeasonality = new ArrayList<ArrayList<ArrayList<ArrayList<Double>>>>();

    private ArrayList<Double> allModelErrorTrend = new ArrayList<Double>();

    private ArrayList<Double> allModelErrorSeasonality = new ArrayList<Double>();

    private double mean = 0;

    private double standardDeviation = 1;

    private ArrayList<Double> rmsErrorTrend = new ArrayList<Double>();

    private ArrayList<Double> rmsErrorSeasonality = new ArrayList<Double>();

    private int outerLoopCount = 0;

    private String modelName = "";

    public HyperParameters() {
    }

    public void setLearningRateUpperLimit(double rate) {
        this.learningRateUpperLimit = rate;
    }

    public double getLearningRateUpperLimit() {
        return this.learningRateUpperLimit;
    }

    public void setLearningRateLowerLimit(double val) {
        this.learnignRateLowerLimit = val;
    }

    public double getLearningRateLowerLimit() {
        return this.learnignRateLowerLimit;
    }

    public void setWiInit(double val) {
        this.wiInit = val;
    }

    public double getWiInit() {
        return this.wiInit;
    }

    public void setWoInit(double val) {
        this.woInit = val;
    }

    public double getWoInit() {
        return this.woInit;
    }

    public void setWzInit(double val) {
        this.wzInit = val;
    }

    public double getWzInit() {
        return this.wzInit;
    }

    public void setriInit(double rate) {
        this.riInit = rate;
    }

    public double getRiInit() {
        return this.riInit;
    }

    public void setRoInit(double val) {
        this.roInit = val;
    }

    public double getRoInit() {
        return this.roInit;
    }

    public void setRzInit(double val) {
        this.rzInit = val;
    }

    public double getRzInit() {
        return this.rzInit;
    }

    public void setYtInit(double val) {
        this.ytInit = val;
    }

    public double getYtInit() {
        return this.ytInit;
    }

    public void setCtInit(double val) {
        this.ctInit = val;
    }

    public double getCtInit() {
        return this.ctInit;
    }

    public void setWfInit(double val) {
        this.wfInit = val;
    }

    public double getWfInit() {
        return this.wfInit;
    }

    public void setRfInit(double val) {
        this.rfInit = val;
    }

    public double getRfInit() {
        return this.rfInit;
    }

    public int getWindowSizeSeasonality() {
        return this.windowSizeSeasonality;
    }

    public int getGdIterration() {
        return this.gdIterration;
    }

    public void setGdIterration(int val) {
        this.gdIterration = val;
    }

    public int getWindowSizeTrend() {
        return this.windowSizeTrend;
    }

    public double getScalingMin() {
        return this.scalingMin;
    }

    public double getScalingMax() {
        return this.scalingMax;
    }

    public void setCount(int val) {
        this.count = val;
    }

    public int getCount() {
        return this.count;
    }

    public void setDatasplitTrain(double val) {
        this.dataSplitTrain = val;
    }

    public double getDataSplitTrain() {
        return this.dataSplitTrain;
    }

    public void setDatasplitValidate(double val) {
        this.dataSplitValidate = val;
    }

    public double getDataSplitValidate() {
        return this.dataSplitValidate;
    }

    public int getTrendPoint() {
        return this.trendPoints;
    }

    public int getEpoch() {
        return this.epoch;
    }

    public int getInterval() {
        return this.interval;
    }

    public void setRmsErrorTrend(double val) {
        this.rmsErrorTrend.add(val);
    }

    public void setRmsErrorSeasonality(double val) {
        this.rmsErrorSeasonality.add(val);
    }

    public ArrayList<Double> getRmsErrorSeasonality() {
        return this.rmsErrorSeasonality;
    }

    public ArrayList<Double> getRmsErrorTrend() {
        return this.rmsErrorTrend;
    }

    public void setEpochTrack(int val) {
        this.epochTrack = val;
    }

    public int getEpochTrack() {
        return this.epochTrack;
    }

    public int getMinimumErrorModelSeasonality() {
        return this.rmsErrorSeasonality.indexOf(Collections.min(this.rmsErrorSeasonality));
    }

    public int getMinimumErrorModelTrend() {
        return this.rmsErrorTrend.indexOf(Collections.min(this.rmsErrorTrend));
    }

    public int getOuterLoopCount() {
        return this.outerLoopCount;
    }

    public void setOuterLoopCount(int val) {
        this.outerLoopCount = val;
    }

    public int getBatchSize() {
        return this.batchSize;
    }

    public int getBatchTrack() {
        return this.batchTrack;
    }

    public void setBatchTrack(int val) {
        this.batchTrack = val;
    }

    public void setModelName(String val) {
        this.modelName = val;
    }

    public String getModelName() {
        return this.modelName;
    }

    public double getMean() {
        return this.mean;
    }

    public double getStandardDeviation() {
        return this.standardDeviation;
    }

    public double getTargetError() {
        return this.targetError;
    }

    public void setTargetError(double val) {
        this.targetError = val;
    }

    public int getMaxItter() {
        return this.maxItterFactor;
    }

    public void updatModelTrend(ArrayList<ArrayList<ArrayList<Double>>> val) {
        this.modelTrend.add(val);
    }

    public ArrayList<ArrayList<ArrayList<Double>>> getlastModelTrend() {
        return this.modelTrend.get(this.modelTrend.size() - 1);
    }

    public ArrayList<ArrayList<ArrayList<Double>>> getBestModelTrend() {
        return this.modelTrend.get(this.getMinimumErrorModelTrend());
    }

    public ArrayList<ArrayList<ArrayList<Double>>> getBestModelSeasonality() {
        return this.modelSeasonality.get(this.getMinimumErrorModelSeasonality());
    }

    public ArrayList<ArrayList<ArrayList<ArrayList<Double>>>> getAllModelsTrend() {
        return this.modelTrend;
    }

    public ArrayList<ArrayList<ArrayList<ArrayList<Double>>>> getAllModelSeasonality() {
        return this.modelSeasonality;
    }

    public void setAllModelErrorTrend(ArrayList<Double> val) {
        this.allModelErrorTrend = val;
    }

    public void setAllModelErrorSeason(ArrayList<Double> val) {
        this.allModelErrorSeasonality = val;
    }

    public ArrayList<Double> getAllModelErrorTrend() {
        return this.allModelErrorTrend;
    }

    public ArrayList<Double> getAllModelErrorSeason() {
        return this.allModelErrorSeasonality;
    }

    public ArrayList<ArrayList<ArrayList<Double>>> getlastModelSeasonality() {
        if (this.modelSeasonality == null) {
            throw new IllegalStateException("modelSeasonality list is not initialized.");
        }
        if (this.modelSeasonality.isEmpty()) {
            throw new IllegalStateException(
                    "modelSeasonality list is empty. Cannot retrieve the last model seasonality.");
        }
        return this.modelSeasonality.get(this.modelSeasonality.size() - 1);
    }

    public void resetModelErrorValue() {
        this.rmsErrorSeasonality = new ArrayList<Double>();
        this.rmsErrorTrend = new ArrayList<Double>();
    }

    public void updateModelSeasonality(ArrayList<ArrayList<ArrayList<Double>>> val) {
        this.modelSeasonality.add(val);
    }

    public void printHyperParameters() {
        var string = new StringBuilder() //
                .append("learningRateUpperLimit=").append(this.learningRateUpperLimit).append("\n") //
                .append("learnignRateLowerLimit=").append(this.learnignRateLowerLimit).append("\n") //
                .append("wiInit=").append(this.wiInit).append("\n") //
                .append("woInit=").append(this.woInit).append("\n") //
                .append("wzInit=").append(this.wzInit).append("\n") //
                .append("riInit=").append(this.riInit).append("\n") //
                .append("roInit=").append(this.roInit).append("\n") //
                .append("rzInit=").append(this.rzInit).append("\n") //
                .append("ytInit=").append(this.ytInit).append("\n") //
                .append("ctInit=").append(this.ctInit).append("\n") //
                .append("Epoch=").append(this.epoch).append("\n") //
                .append("windowSizeSeasonality=").append(this.windowSizeSeasonality).append("\n") //
                .append("windowSizeTrend=").append(this.windowSizeTrend).append("\n") //
                .append("scalingMin=").append(this.scalingMin).append("\n") //
                .append("scalingMax=").append(this.scalingMax).append("\n") //
                .append("RMS error trend=").append(this.getRmsErrorTrend()).append("\n") //
                .append("RMS error seasonality=").append(this.getRmsErrorSeasonality()).append("\n") //
                .append("Count value=").append(this.count).append("\n") //
                .append("Outer loop Count=").append(this.outerLoopCount).append("\n") //
                .append("Epoch track=").append(this.epochTrack).append("\n") //
                .toString();

        System.out.println(string);
    }

    public void update() {
        int minErrorIndTrend = this.getMinimumErrorModelTrend();
        int minErrorIndSeasonlity = this.getMinimumErrorModelSeasonality();

        // uipdating models
        var modelTrendTemp = this.modelTrend.get(minErrorIndTrend);
        final var modelTempSeasonality = this.modelSeasonality.get(minErrorIndSeasonlity);
        this.modelTrend.clear();
        this.modelSeasonality.clear();
        this.modelTrend.add(modelTrendTemp);
        this.modelSeasonality.add(modelTempSeasonality);

        // updating index
        double minErrorTrend = this.rmsErrorTrend.get(minErrorIndTrend);
        final double minErrorSeasonality = this.rmsErrorSeasonality.get(minErrorIndSeasonlity);
        this.rmsErrorTrend.clear();
        this.rmsErrorSeasonality.clear();
        this.rmsErrorTrend.add(minErrorTrend);
        this.rmsErrorSeasonality.add(minErrorSeasonality);
        this.count = 1;
        this.lastTrainedDate = OffsetDateTime.now();
    }

}
