// Sources/ATenCXXDoctests/group_norm_doctests.cpp
#include "include/doctest.h"
#include "tensor_shim.hpp"

#include <array>
#include <cmath>
#include <vector>

using i64 = int64_t;

namespace {
TTSTensor makeTensorF64(const std::vector<double> &host, const std::vector<i64> &shape) {
  return TTSTensor::fromArray(host, shape);
}

i64 flatIndex(i64 n, i64 c, i64 h, i64 w, i64 N, i64 C, i64 H, i64 W) {
  return (((n * C + c) * H) + h) * W + w;
}

struct Layout {
  i64 N{1};
  i64 C{4};
  i64 H{2};
  i64 W{2};
  i64 groups{2};
  i64 channelsPerGroup{C / groups};
  i64 spatialCount{H * W};
  i64 elementsPerGroup{channelsPerGroup * spatialCount};
};

std::tuple<TTSTensor, TTSTensor, TTSTensor> nativeGroupNormForward(
  const std::vector<double> &inputHost,
  const std::vector<double> &gammaHost,
  const std::vector<double> &betaHost,
  const Layout &layout,
  double eps)
{
  TTSTensor input = makeTensorF64(inputHost, {layout.N, layout.C, layout.H, layout.W});
  TTSTensor gamma = makeTensorF64(gammaHost, {layout.C});
  TTSTensor beta = makeTensorF64(betaHost, {layout.C});
  return TTSTensor::_native_group_norm_forward(
    input,
    layout.groups,
    &gamma,
    &beta,
    eps);
}

double forwardSum(
  const std::vector<double> &inputHost,
  const std::vector<double> &gammaHost,
  const std::vector<double> &betaHost,
  const Layout &layout,
  double eps)
{
  auto tuple = nativeGroupNormForward(inputHost, gammaHost, betaHost, layout, eps);
  TTSTensor y = TTSTensor::_native_group_norm_forward_get0(tuple);
  TTSTensor total = y.sumAll();
  return total._t().item<double>();
}
}

DOCTEST_TEST_CASE("group_norm forward matches manual computation (NCHW)") {
  const double eps = 1e-5;
  Layout layout;

  std::vector<double> inputHost{
      0.0, 1.0, 2.0, 3.0,
      4.0, 5.0, 6.0, 7.0,
      8.0, 9.0, 10.0, 11.0,
      12.0, 13.0, 14.0, 15.0};

  std::vector<double> gammaHost{1.5, 0.5, -1.0, 2.0};
  std::vector<double> betaHost{0.1, -0.2, 0.0, 0.25};

  auto tuple = nativeGroupNormForward(inputHost, gammaHost, betaHost, layout, eps);
  TTSTensor y = TTSTensor::_native_group_norm_forward_get0(tuple);
  TTSTensor meanT = TTSTensor::_native_group_norm_forward_get1(tuple);
  TTSTensor rstdT = TTSTensor::_native_group_norm_forward_get2(tuple);

  std::vector<double> mean(layout.groups, 0.0);
  for (i64 c = 0; c < layout.C; ++c) {
    const i64 g = c / layout.channelsPerGroup;
    for (i64 h = 0; h < layout.H; ++h) {
      for (i64 w = 0; w < layout.W; ++w) {
        mean[g] += inputHost[flatIndex(0, c, h, w, layout.N, layout.C, layout.H, layout.W)];
      }
    }
  }
  for (double &m : mean) {
    m /= static_cast<double>(layout.elementsPerGroup);
  }

  std::vector<double> var(layout.groups, 0.0);
  for (i64 c = 0; c < layout.C; ++c) {
    const i64 g = c / layout.channelsPerGroup;
    for (i64 h = 0; h < layout.H; ++h) {
      for (i64 w = 0; w < layout.W; ++w) {
        const double val = inputHost[flatIndex(0, c, h, w, layout.N, layout.C, layout.H, layout.W)];
        const double diff = val - mean[g];
        var[g] += diff * diff;
      }
    }
  }
  for (double &v : var) {
    v /= static_cast<double>(layout.elementsPerGroup);
  }

  std::vector<double> normHost(inputHost.size());
  for (i64 c = 0; c < layout.C; ++c) {
    const i64 g = c / layout.channelsPerGroup;
    const double denom = std::sqrt(var[g] + eps);
    for (i64 h = 0; h < layout.H; ++h) {
      for (i64 w = 0; w < layout.W; ++w) {
        const auto idx = flatIndex(0, c, h, w, layout.N, layout.C, layout.H, layout.W);
        normHost[idx] = (inputHost[idx] - mean[g]) / denom;
      }
    }
  }

  std::vector<double> outHost(inputHost.size());
  for (i64 c = 0; c < layout.C; ++c) {
    const double gammaVal = gammaHost[static_cast<size_t>(c)];
    const double betaVal = betaHost[static_cast<size_t>(c)];
    for (i64 h = 0; h < layout.H; ++h) {
      for (i64 w = 0; w < layout.W; ++w) {
        const auto idx = flatIndex(0, c, h, w, layout.N, layout.C, layout.H, layout.W);
        outHost[idx] = normHost[idx] * gammaVal + betaVal;
      }
    }
  }

  std::vector<double> rstd(layout.groups);
  for (i64 g = 0; g < layout.groups; ++g) {
    rstd[g] = 1.0 / std::sqrt(var[g] + eps);
  }

  TTSTensor expectedY = makeTensorF64(outHost, {layout.N, layout.C, layout.H, layout.W});
  TTSTensor expectedMean = makeTensorF64(mean, {layout.N, layout.groups});
  TTSTensor expectedRstd = makeTensorF64(rstd, {layout.N, layout.groups});

  DOCTEST_CHECK(y.allclose(expectedY, 1e-10, 0.0, false));
  DOCTEST_CHECK(meanT.allclose(expectedMean, 1e-12, 0.0, false));
  DOCTEST_CHECK(rstdT.allclose(expectedRstd, 1e-12, 0.0, false));
}

DOCTEST_TEST_CASE("group_norm backward matches finite differences") {
  const double eps = 1e-5;
  const double h = 1e-4;
  Layout layout;

  std::vector<double> inputHost{
      0.0, 1.0, 2.0, 3.0,
      4.0, 5.0, 6.0, 7.0,
      8.0, 9.0, 10.0, 11.0,
      12.0, 13.0, 14.0, 15.0};

  std::vector<double> gammaHost{1.0, 1.0, 1.0, 1.0};
  std::vector<double> betaHost{0.0, 0.0, 0.0, 0.0};
  std::vector<double> gradOutHost(inputHost.size(), 1.0);

  // Base forward for backward inputs
  TTSTensor inputBase = makeTensorF64(inputHost, {layout.N, layout.C, layout.H, layout.W});
  TTSTensor gammaBase = makeTensorF64(gammaHost, {layout.C});
  TTSTensor betaBase = makeTensorF64(betaHost, {layout.C});
  auto forwardTuple = TTSTensor::_native_group_norm_forward(
      inputBase,
      layout.groups,
      &gammaBase,
      &betaBase,
      eps);

  TTSTensor meanT = TTSTensor::_native_group_norm_forward_get1(forwardTuple);
  TTSTensor rstdT = TTSTensor::_native_group_norm_forward_get2(forwardTuple);

  TTSTensor gradOut = makeTensorF64(gradOutHost, {layout.N, layout.C, layout.H, layout.W});
  auto backwardTuple = TTSTensor::_native_group_norm_backward(
      gradOut,
      inputBase,
      meanT,
      rstdT,
      layout.groups,
      &gammaBase);

  TTSTensor gradInput = TTSTensor::_native_group_norm_backward_get0(backwardTuple);
  TTSTensor gradWeight = TTSTensor::_native_group_norm_backward_get1(backwardTuple);
  TTSTensor gradBias = TTSTensor::_native_group_norm_backward_get2(backwardTuple);

  std::vector<double> numericGradInput(inputHost.size());
  for (size_t idx = 0; idx < inputHost.size(); ++idx) {
    std::vector<double> plus = inputHost;
    std::vector<double> minus = inputHost;
    plus[idx] += h;
    minus[idx] -= h;
    const double fPlus = forwardSum(plus, gammaHost, betaHost, layout, eps);
    const double fMinus = forwardSum(minus, gammaHost, betaHost, layout, eps);
    numericGradInput[idx] = (fPlus - fMinus) / (2.0 * h);
  }

  std::vector<double> numericGradWeight(gammaHost.size());
  for (size_t idx = 0; idx < gammaHost.size(); ++idx) {
    std::vector<double> plus = gammaHost;
    std::vector<double> minus = gammaHost;
    plus[idx] += h;
    minus[idx] -= h;
    const double fPlus = forwardSum(inputHost, plus, betaHost, layout, eps);
    const double fMinus = forwardSum(inputHost, minus, betaHost, layout, eps);
    numericGradWeight[idx] = (fPlus - fMinus) / (2.0 * h);
  }

  std::vector<double> numericGradBias(betaHost.size());
  for (size_t idx = 0; idx < betaHost.size(); ++idx) {
    std::vector<double> plus = betaHost;
    std::vector<double> minus = betaHost;
    plus[idx] += h;
    minus[idx] -= h;
    const double fPlus = forwardSum(inputHost, gammaHost, plus, layout, eps);
    const double fMinus = forwardSum(inputHost, gammaHost, minus, layout, eps);
    numericGradBias[idx] = (fPlus - fMinus) / (2.0 * h);
  }

  TTSTensor expectedGradInput = makeTensorF64(numericGradInput, {layout.N, layout.C, layout.H, layout.W});
  TTSTensor expectedGradWeight = makeTensorF64(numericGradWeight, {layout.C});
  TTSTensor expectedGradBias = makeTensorF64(numericGradBias, {layout.C});

  DOCTEST_CHECK(gradInput.allclose(expectedGradInput, 1e-5, 1e-7, false));
  DOCTEST_CHECK(gradWeight.allclose(expectedGradWeight, 1e-6, 1e-7, false));
  DOCTEST_CHECK(gradBias.allclose(expectedGradBias, 1e-6, 1e-7, false));
}
