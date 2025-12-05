package predictor.lstm.data;

import java.time.OffsetDateTime;
import java.util.ArrayList;

/**
 * A record to hold time series data, consisting of a list of dates and a corresponding list of values.
 */
public record TimeSeriesData(ArrayList<OffsetDateTime> dates, ArrayList<Double> values) {
}
