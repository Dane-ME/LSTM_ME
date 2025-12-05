package predictor.lstm.preprocessingpipeline;

public interface PipelineInterface<O, I> {

    /**
     * Adds a stage to the pipeline.
     *
     * @param stage The stage to be added to the pipeline.
     */
    public void add(Stage<O, I> stage);

    /**
     * Executes the pipeline, processing the data through the added stages.
     *
     * @return The result of executing the pipeline.
     */
    public Object execute();
}