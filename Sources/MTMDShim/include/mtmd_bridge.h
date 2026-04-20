#ifndef MTMD_BRIDGE_H
#define MTMD_BRIDGE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// This is a way to declare types and functions that are available in the
// xcframework as symbols but there is no header file for them as currently the
// mtmd headers contain c++ code/constructs that cause clang to be unable to
// import them (it would fail to import the llama module if those headers are
// included.
// mtmd is still experimental but this gives users a compromise were they can
// create their own header file like this one, which is in it's own module, and
// then be able to import this module and use the mtmd symbols.

struct mtmd_decoder_pos {
    uint32_t t;
    uint32_t x;
    uint32_t y;
    uint32_t z;
};

const char * mtmd_default_marker(void);

#ifdef __cplusplus
}
#endif

#endif
