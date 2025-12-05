package predictor.lstm.util.forgetgate;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.stream.IntStream;
import predictor.lstm.util.MatrixWeight;
import predictor.lstm.util.improved.AdaptiveLearningRateImproved;

public class LstmWithForgetGate {

    private double[] inputData;
    private double outputData;
    
    // Gradients
    private double derivativeLWrtRi = 0;
    private double derivativeLWrtRo = 0;
    private double derivativeLWrtRz = 0;
    private double derivativeLWrtRf = 0; // NEW
    
    private double derivativeLWrtWi = 0;
    private double derivativeLWrtWo = 0;
    private double derivativeLWrtWz = 0;
    private double derivativeLWrtWf = 0; // NEW
    
    // Accumulated Squared Gradients for Adagrad
    // 0:Wi, 1:Wo, 2:Wz, 3:Ri, 4:Ro, 5:Rz, 6:Wf, 7:Rf
    private ArrayList<Double> accumulatedGradientSq; 

    private double learningRate;
    private int epoch = 100;
    private double epsilon = 1e-8;

    protected ArrayList<CellWithForgetGate> cells;

    public LstmWithForgetGate(LstmBuilder builder) {
        this.inputData = builder.inputData;
        this.outputData = builder.outputData;
        this.learningRate = builder.learningRate;
        this.epoch = builder.epoch;
        
        // Initialize Adagrad state (8 parameters)
        this.accumulatedGradientSq = new ArrayList<>();
        for(int i=0; i<8; i++) {
            this.accumulatedGradientSq.add(0.0);
        }
    }

    /**
     * Forward propagation.
     */
    public void forwardprop(boolean training) {
        try {
            double dropoutRate = 0.2; 

            for (int i = 0; i < this.cells.size(); i++) {
                if (training) {
                    this.cells.get(i).setDropoutEnabled(true);
                    this.cells.get(i).setDropoutRate(dropoutRate);
                } else {
                    this.cells.get(i).setDropoutEnabled(false);
                }

                this.cells.get(i).forwardPropogation();

                if (i < this.cells.size() - 1) {
                    this.cells.get(i + 1).setYtMinusOne(this.cells.get(i).getYt());
                    this.cells.get(i + 1).setCtMinusOne(this.cells.get(i).getCt());
                    
                     this.cells.get(i).setError(
                            this.cells.get(i).getYt() - this.cells.get(i + 1).getXt());
                } else {
                     this.cells.get(i).setError(this.cells.get(i).getYt() - this.outputData);
                }
            }
        } catch (IndexOutOfBoundsException e) {
            e.printStackTrace();
        }
    }

    /**
     * Backward propagation.
     */
    public void backwardprop() {
        // Reset gradients
        this.derivativeLWrtRi = 0;
        this.derivativeLWrtRo = 0;
        this.derivativeLWrtRz = 0;
        this.derivativeLWrtRf = 0;
        
        this.derivativeLWrtWi = 0;
        this.derivativeLWrtWo = 0;
        this.derivativeLWrtWz = 0;
        this.derivativeLWrtWf = 0;

        for (int i = this.cells.size() - 1; i >= 0; i--) {
            if (i < this.cells.size() - 1) {
                // Pass gradient of Cell State from future to past
                // With Forget Gate: dL/dc_{t} = ... + dL/dc_{t+1} * f_{t+1}
                double nextDlByDc = this.cells.get(i + 1).getDlByDc();
                double nextFt = this.cells.get(i + 1).getFt();
                this.cells.get(i).setDlByDc(nextDlByDc * nextFt);
            }
            this.cells.get(i).backwardPropogation();
        }

        // Accumulate gradients for weights
        for (int i = 0; i < this.cells.size(); i++) {
            this.derivativeLWrtRi += this.cells.get(i).getYtMinusOne() * this.cells.get(i).getDelI();
            this.derivativeLWrtRo += this.cells.get(i).getYtMinusOne() * this.cells.get(i).getDelO();
            this.derivativeLWrtRz += this.cells.get(i).getYtMinusOne() * this.cells.get(i).getDelZ();
            this.derivativeLWrtRf += this.cells.get(i).getYtMinusOne() * this.cells.get(i).getDelF();

            this.derivativeLWrtWi += this.cells.get(i).getXt() * this.cells.get(i).getDelI();
            this.derivativeLWrtWo += this.cells.get(i).getXt() * this.cells.get(i).getDelO();
            this.derivativeLWrtWz += this.cells.get(i).getXt() * this.cells.get(i).getDelZ();
            this.derivativeLWrtWf += this.cells.get(i).getXt() * this.cells.get(i).getDelF();
        }
        
        double n = this.cells.size();
        var gradients = new ArrayList<Double>();
        gradients.add(this.derivativeLWrtWi / n);
        gradients.add(this.derivativeLWrtWo / n);
        gradients.add(this.derivativeLWrtWz / n);
        gradients.add(this.derivativeLWrtRi / n);
        gradients.add(this.derivativeLWrtRo / n);
        gradients.add(this.derivativeLWrtRz / n);
        // Add new gradients
        gradients.add(this.derivativeLWrtWf / n);
        gradients.add(this.derivativeLWrtRf / n);

        this.updateweights(gradients);
    }

    protected void updateweights(ArrayList<Double> gradients) {
        var optimizer = new AdaptiveLearningRateImproved();

        // Update accumulated squared gradients
        for(int k=0; k<8; k++) {
            double g = gradients.get(k);
            double newG = this.accumulatedGradientSq.get(k) + g*g;
            this.accumulatedGradientSq.set(k, newG);
        }

        // Current weights (take from first cell)
        double curWi = this.cells.get(0).getWi();
        double curWo = this.cells.get(0).getWo();
        double curWz = this.cells.get(0).getWz();
        double curRi = this.cells.get(0).getRi();
        double curRo = this.cells.get(0).getRo();
        double curRz = this.cells.get(0).getRz();
        double curWf = this.cells.get(0).getWf();
        double curRf = this.cells.get(0).getRf();
        
        double newWi = optimizer.adagradUpdate(curWi, gradients.get(0), this.accumulatedGradientSq.get(0), learningRate, epsilon);
        double newWo = optimizer.adagradUpdate(curWo, gradients.get(1), this.accumulatedGradientSq.get(1), learningRate, epsilon);
        double newWz = optimizer.adagradUpdate(curWz, gradients.get(2), this.accumulatedGradientSq.get(2), learningRate, epsilon);
        double newRi = optimizer.adagradUpdate(curRi, gradients.get(3), this.accumulatedGradientSq.get(3), learningRate, epsilon);
        double newRo = optimizer.adagradUpdate(curRo, gradients.get(4), this.accumulatedGradientSq.get(4), learningRate, epsilon);
        double newRz = optimizer.adagradUpdate(curRz, gradients.get(5), this.accumulatedGradientSq.get(5), learningRate, epsilon);
        double newWf = optimizer.adagradUpdate(curWf, gradients.get(6), this.accumulatedGradientSq.get(6), learningRate, epsilon);
        double newRf = optimizer.adagradUpdate(curRf, gradients.get(7), this.accumulatedGradientSq.get(7), learningRate, epsilon);

        for (int i = 0; i < this.cells.size(); i++) {
            this.cells.get(i).setWi(newWi);
            this.cells.get(i).setWo(newWo);
            this.cells.get(i).setWz(newWz);
            this.cells.get(i).setRi(newRi);
            this.cells.get(i).setRo(newRo);
            this.cells.get(i).setRz(newRz);
            this.cells.get(i).setWf(newWf);
            this.cells.get(i).setRf(newRf);
        }
    }

    public ArrayList<ArrayList<Double>> train() {
        
        var historyWi = new ArrayList<ArrayList<Double>>();
        var historyWo = new ArrayList<ArrayList<Double>>();
        var historyWz = new ArrayList<ArrayList<Double>>();
        var historyRi = new ArrayList<ArrayList<Double>>();
        var historyRo = new ArrayList<ArrayList<Double>>();
        var historyRz = new ArrayList<ArrayList<Double>>();
        var historyWf = new ArrayList<ArrayList<Double>>();
        var historyRf = new ArrayList<ArrayList<Double>>();
        var historyYt = new ArrayList<ArrayList<Double>>();
        var historyCt = new ArrayList<ArrayList<Double>>();
        var errorList = new ArrayList<Double>();

        for (int i = 0; i < this.epoch; i++) {
            this.forwardprop(true);
            this.backwardprop();

            var wiList = new ArrayList<Double>();
            var woList = new ArrayList<Double>();
            var wzList = new ArrayList<Double>();
            var riList = new ArrayList<Double>();
            var roList = new ArrayList<Double>();
            var rzList = new ArrayList<Double>();
            var wfList = new ArrayList<Double>();
            var rfList = new ArrayList<Double>();
            var ytList = new ArrayList<Double>();
            var ctList = new ArrayList<Double>();

            for (int j = 0; j < this.cells.size(); j++) {
                wiList.add(this.cells.get(j).getWi());
                woList.add(this.cells.get(j).getWo());
                wzList.add(this.cells.get(j).getWz());
                riList.add(this.cells.get(j).getRi());
                roList.add(this.cells.get(j).getRo());
                rzList.add(this.cells.get(j).getRz());
                wfList.add(this.cells.get(j).getWf());
                rfList.add(this.cells.get(j).getRf());
                ytList.add(this.cells.get(j).getYt());
                ctList.add(this.cells.get(j).getCt());
            }
            
            errorList.add(this.cells.get(this.cells.size() - 1).getError());
            
            historyWi.add(wiList);
            historyWo.add(woList);
            historyWz.add(wzList);
            historyRi.add(riList);
            historyRo.add(roList);
            historyRz.add(rzList);
            historyWf.add(wfList);
            historyRf.add(rfList);
            historyYt.add(ytList);
            historyCt.add(ctList);
        }

        int globalMinimaIndex = findGlobalMinima(errorList);

        var returnArray = new ArrayList<ArrayList<Double>>();

        returnArray.add(historyWi.get(globalMinimaIndex));
        returnArray.add(historyWo.get(globalMinimaIndex));
        returnArray.add(historyWz.get(globalMinimaIndex));
        returnArray.add(historyRi.get(globalMinimaIndex));
        returnArray.add(historyRo.get(globalMinimaIndex));
        returnArray.add(historyRz.get(globalMinimaIndex));
        returnArray.add(historyYt.get(globalMinimaIndex));
        returnArray.add(historyCt.get(globalMinimaIndex));
        // Append new weights at the end
        returnArray.add(historyWf.get(globalMinimaIndex)); // Index 8
        returnArray.add(historyRf.get(globalMinimaIndex)); // Index 9

        return returnArray;
    }

    public static int findGlobalMinima(ArrayList<Double> data) {
        return IntStream.range(0, data.size())
                .boxed()
                .min(Comparator.comparingDouble(i -> Math.abs(data.get(i))))
                .orElse(-1);
    }

    // Setters need to handle the new indices (8 and 9)
    public synchronized void setWi(ArrayList<ArrayList<Double>> val) {
        for (int i = 0; i < val.get(0).size(); i++) { this.cells.get(i).setWi(val.get(0).get(i)); }
    }
    public synchronized void setWo(ArrayList<ArrayList<Double>> val) {
        for (int i = 0; i < val.get(1).size(); i++) { this.cells.get(i).setWo(val.get(1).get(i)); }
    }
    public synchronized void setWz(ArrayList<ArrayList<Double>> val) {
        for (int i = 0; i < val.get(2).size(); i++) { this.cells.get(i).setWz(val.get(2).get(i)); }
    }
    public synchronized void setRi(ArrayList<ArrayList<Double>> val) {
        for (int i = 0; i < val.get(3).size(); i++) { this.cells.get(i).setRi(val.get(3).get(i)); }
    }
    public synchronized void setRo(ArrayList<ArrayList<Double>> val) {
        for (int i = 0; i < val.get(4).size(); i++) { this.cells.get(i).setRo(val.get(4).get(i)); }
    }
    public synchronized void setRz(ArrayList<ArrayList<Double>> val) {
        for (int i = 0; i < val.get(5).size(); i++) { this.cells.get(i).setRz(val.get(5).get(i)); }
    }
    public synchronized void setYt(ArrayList<ArrayList<Double>> val) {
        for (int i = 0; i < val.get(6).size(); i++) { this.cells.get(i).setYt(val.get(6).get(i)); }
    }
    public synchronized void setCt(ArrayList<ArrayList<Double>> val) {
        for (int i = 0; i < val.get(7).size(); i++) { this.cells.get(i).setCt(val.get(7).get(i)); }
    }
    // New setters
    public synchronized void setWf(ArrayList<ArrayList<Double>> val) {
        if (val.size() > 8) {
            for (int i = 0; i < val.get(8).size(); i++) { this.cells.get(i).setWf(val.get(8).get(i)); }
        }
    }
    public synchronized void setRf(ArrayList<ArrayList<Double>> val) {
        if (val.size() > 9) {
            for (int i = 0; i < val.get(9).size(); i++) { this.cells.get(i).setRf(val.get(9).get(i)); }
        }
    }

    public synchronized void initilizeCells() {
        this.cells = new ArrayList<>();
        for (int i = 0; i < this.inputData.length; i++) {
            CellWithForgetGate cell = new CellWithForgetGate(this.inputData[i], this.outputData);
            this.cells.add(cell);
        }
    }

    public static class LstmBuilder {
        protected double[] inputData;
        protected double outputData;
        protected double learningRate;
        protected int epoch = 100;

        public LstmBuilder(double[] inputData, double outputData) {
            this.inputData = inputData;
            this.outputData = outputData;
        }

        public LstmBuilder setInputData(double[] inputData) { this.inputData = inputData; return this; }
        public LstmBuilder setOutputData(double outputData) { this.outputData = outputData; return this; }
        public LstmBuilder setLearningRate(double learningRate) { this.learningRate = learningRate; return this; }
        public LstmBuilder setEpoch(int epoch) { this.epoch = epoch; return this; }

        public LstmWithForgetGate build() {
            return new LstmWithForgetGate(this);
        }
    }
}
