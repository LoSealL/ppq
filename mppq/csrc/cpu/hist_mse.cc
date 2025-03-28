# include "hist_mse.h"
# include <algorithm>
# include <numeric>

float compute_mse_loss(
    const vector<int64_t> &hist,
    const int start,
    const int step,
    const int end) {
    float loss = 0.0f;
    int64_t num_of_elements = std::accumulate(hist.begin(), hist.end(), 0);
    for(size_t idx = 0; idx < hist.size(); idx++) {
        float error = 0.0f;
        int64_t bin = hist[idx];
        if (idx < start) {
            error = static_cast<float>(start - idx - 1) + 0.5f;
        } else if(idx > end) {
            error = static_cast<float>(idx - end) + 0.5f;
        } else {
            int64_t l_idx = (idx - start) % step;
            int64_t r_idx = step - l_idx - 1;
            if (l_idx == r_idx) {
                error = static_cast<float>(l_idx) + 0.25f;
            } else {
                float l_err = l_idx + 0.5f;
                float r_err = r_idx + 0.5f;
                error = std::min(l_err, r_err);
            }
        }
        loss += (bin * error * error) / num_of_elements;
    }
    return loss;
}
