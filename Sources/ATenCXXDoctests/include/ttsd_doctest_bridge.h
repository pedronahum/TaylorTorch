#pragma once
#ifdef __cplusplus
extern "C" {
#endif

/// Runs the embedded doctest suite and returns the doctest exit code (0 = pass).
int ttsd_run_doctests(void);

#ifdef __cplusplus
}
#endif
