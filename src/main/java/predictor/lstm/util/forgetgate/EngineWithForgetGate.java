package predictor.lstm.util.forgetgate;

import static predictor.lstm.preprocessing.DataModification.scaleBack;
import static predictor.lstm.utilities.MathUtils.sigmoid;
import static predictor.lstm.utilities.MathUtils.tanh;
import static predictor.lstm.utilities.UtilityConversion.getMinIndex;

import java.util.ArrayList;

import predictor.lstm.common.DataStatistics;
import predictor.lstm.common.HyperParameters;
import predictor.lstm.util.forgetgate.LstmWithForgetGate.LstmBuilder;

public class EngineWithForgetGate {

    private double[][] inputMatrix;
    private double[] targetVector;
    private double[][] validateData;
    private double[] validateTarget;
    private double learningRate;

    public double getLearningRate() { return this.learningRate; }

    private final ArrayList<ArrayList<ArrayList<Double>>> weights = new ArrayList<ArrayList<ArrayList<Double>>>();
    private final ArrayList<ArrayList<Double>> finalWeights = new ArrayList<ArrayList<Double>>();

    public void fit(int epochs, ArrayList<ArrayList<Double>> val, HyperParameters hyperParameters) {
        var rate = new predictor.lstm.util.improved.AdaptiveLearningRateImproved();

        this.learningRate = rate.scheduler(hyperParameters);

        var ls = new LstmBuilder(this.inputMatrix[0], this.targetVector[0])
                .setLearningRate(this.learningRate)
                .setEpoch(epochs)
                .build();

        ls.initilizeCells();

        // Standard
        ls.setWi(val); ls.setWo(val); ls.setWz(val);
        ls.setRi(val); ls.setRo(val); ls.setRz(val);
        ls.setYt(val); ls.setCt(val);
        
        // Forget Gate - Check if val has enough elements (indices 8 and 9)
        // If val comes from initialization or previous training of ForgetGate model, it should have 10 elements.
        if (val.size() > 8) {
            ls.setWf(val);
            ls.setRf(val);
        }

        var wieghtMatrix = ls.train();
        this.weights.add(wieghtMatrix);

        for (int i = 1; i < this.inputMatrix.length; i++) {
            this.learningRate = rate.scheduler(hyperParameters);
            ls = new LstmBuilder(this.inputMatrix[i], this.targetVector[i])
                    .setLearningRate(this.learningRate)
                    .setEpoch(epochs)
                    .build();

            ls.initilizeCells();

            // Set weights from previous step (wieghtMatrix)
            for (int j = 0; j < ls.cells.size(); j++) {
                ls.cells.get(j).setWi(wieghtMatrix.get(0).get(j));
                ls.cells.get(j).setWo(wieghtMatrix.get(1).get(j));
                ls.cells.get(j).setWz(wieghtMatrix.get(2).get(j));
                ls.cells.get(j).setRi(wieghtMatrix.get(3).get(j));
                ls.cells.get(j).setRo(wieghtMatrix.get(4).get(j));
                ls.cells.get(j).setRz(wieghtMatrix.get(5).get(j));
                ls.cells.get(j).setYtMinusOne(wieghtMatrix.get(6).get(j));
                ls.cells.get(j).setCtMinusOne(wieghtMatrix.get(7).get(j));
                // Forget Gate
                if (wieghtMatrix.size() > 8) {
                    ls.cells.get(j).setWf(wieghtMatrix.get(8).get(j));
                    ls.cells.get(j).setRf(wieghtMatrix.get(9).get(j));
                }
            }

            wieghtMatrix = ls.train();
            this.weights.add(wieghtMatrix);
        }
    }

    // Prediction and Validation methods logic remains similar but must handle extra weights?
    // Actually, Engine is mostly used for Training.
    // LstmPredictor handles prediction.
    // But validation uses this? Let's check Engine.validate() in original code.
    
    // EngineImproved.validate logic:
    // result[i] = this.singleValuePredict(inputData[i], val.get(0), ... val.get(7), ...)
    
    // We need singleValuePredict to support Forget Gate.

    public double[] validate(double[][] inputData, double[] target, ArrayList<ArrayList<Double>> val,
                             HyperParameters hyperParameter) {
        var result = new double[inputData.length];
        for (int i = 0; i < inputData.length; i++) {
            // Need to pass Wf and Rf if available
            ArrayList<Double> wf = (val.size() > 8) ? val.get(8) : null;
            ArrayList<Double> rf = (val.size() > 9) ? val.get(9) : null;
            
            result[i] = this.singleValuePredict(inputData[i], 
                    val.get(0), val.get(1), val.get(2), 
                    val.get(3), val.get(4), val.get(5), 
                    wf, rf,
                    val.get(6), val.get(7), 
                    hyperParameter);
        }
        return result;
    }

    private double singleValuePredict(double[] inputData, 
                                      ArrayList<Double> wi, ArrayList<Double> wo, ArrayList<Double> wz, 
                                      ArrayList<Double> Ri, ArrayList<Double> Ro, ArrayList<Double> Rz,
                                      ArrayList<Double> Wf, ArrayList<Double> Rf, // New
                                      ArrayList<Double> ytV, ArrayList<Double> ctV, 
                                      HyperParameters hyperParameter) {
        var ct = 0.;
        var ctMinusOne = 0.;
        var yt = 0.;
        var standData = inputData;

        for (int i = 0; i < wi.size(); i++) {
            ctMinusOne = ctV.get(i);
            double it = sigmoid(wi.get(i) * standData[i] + Ri.get(i) * yt);
            double ot = sigmoid(wo.get(i) * standData[i] + Ro.get(i) * yt);
            double zt = tanh(wz.get(i) * standData[i] + Rz.get(i) * yt);
            
            // Forget Gate Logic
            double ft = 1.0;
            if (Wf != null && Rf != null) {
                ft = sigmoid(Wf.get(i) * standData[i] + Rf.get(i) * yt);
            }
            
            ct = ft * ctMinusOne + it * zt; // Modified update
            yt = ot * tanh(ct);
        }
        return scaleBack(yt, hyperParameter.getScalingMin(), hyperParameter.getScalingMax());
    }

    public EngineWithForgetGate(EngineBuilder builder) {
        this.inputMatrix = builder.inputMatrix;
        this.targetVector = builder.targetVector;
        this.validateData = builder.validateData;
        this.validateTarget = builder.validateTarget;
    }

    public static class EngineBuilder {
        private double[][] inputMatrix;
        private double[] targetVector;
        private double[][] validateData;
        private double[] validateTarget;

        public EngineBuilder() {}

        public EngineBuilder setInputMatrix(double[][] inputMatrix) {
            this.inputMatrix = inputMatrix;
            return this;
        }

        public EngineBuilder setTargetVector(double[] targetVector) {
            this.targetVector = targetVector;
            return this;
        }

        public EngineBuilder setValidateData(double[][] validateData) {
            this.validateData = validateData;
            return this;
        }

        public EngineBuilder setValidateTarget(double[] validateTarget) {
            this.validateTarget = validateTarget;
            return this;
        }

        public EngineWithForgetGate build() {
            return new EngineWithForgetGate(this);
        }
    }

    public ArrayList<ArrayList<ArrayList<Double>>> getWeights() {
        return this.weights;
    }
}
