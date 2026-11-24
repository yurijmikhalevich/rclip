// Sets the LibRaw processing parameters that have no setter in LibRaw's C API.
#include "libraw/libraw.h"

// Matches rawpy's `postprocess(half_size=True)` with its default parameters, which pass the
// gamma curve as (1 / 2.222, 4.5).
extern "C" void rclip_libraw_use_half_size(libraw_data_t *data) {
  data->params.half_size = 1;
  data->params.gamm[0] = 1.0 / 2.222;
  data->params.gamm[1] = 4.5;
}
