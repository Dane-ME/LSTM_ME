package predictor.lstm.preprocessingpipeline;

import static predictor.lstm.preprocessing.DataModification.groupDataByHourAndMinute;
import static predictor.lstm.utilities.UtilityConversion.to1DArrayList;
import static predictor.lstm.utilities.UtilityConversion.to3DArray;

import java.time.OffsetDateTime;
import java.util.ArrayList;

import predictor.lstm.common.HyperParameters;

public class GroupbyPipe implements Stage<Object, Object> {

    private ArrayList<OffsetDateTime> dates;

    public GroupbyPipe(HyperParameters hype, ArrayList<OffsetDateTime> date) {
        this.dates = date;
    }

    @Override
    public Object execute(Object input) {
        if (input instanceof double[] in) {
            var inList = to1DArrayList(in);
            var groupedByHourAndMinuteList = groupDataByHourAndMinute(inList, this.dates);
            return to3DArray(groupedByHourAndMinuteList);
        } else {
            throw new IllegalArgumentException("Input must be an instance of double[]");
        }
    }
}