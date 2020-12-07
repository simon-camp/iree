func @conv2d_nopadding() attributes { iree.module.export } {
  %inputs = iree.unfoldable_constant dense<[[
      [[ 1.0,  2.0], [ 3.0,  4.0], [ 5.0,  6.0], [ 7.0,  8.0], [ 9.0, 10.0]],
      [[11.0, 12.0], [13.0, 14.0], [15.0, 16.0], [17.0, 18.0], [19.0, 20.0]],
      [[21.0, 22.0], [23.0, 24.0], [25.0, 26.0], [27.0, 28.0], [29.0, 30.0]],
      [[31.0, 32.0], [33.0, 34.0], [35.0, 36.0], [37.0, 38.0], [39.0, 40.0]]]]> : tensor<1x4x5x2xf32>
  %weights = iree.unfoldable_constant dense<[
      [[[ 1.0], [ 2.0]], [[ 3.0], [ 4.0]]],
      [[[ 5.0], [ 6.0]], [[ 7.0], [ 8.0]]],
      [[[ 9.0], [10.0]], [[11.0], [12.0]]]]> : tensor<3x2x2x1xf32>
  %res = "mhlo.convolution"(%inputs, %weights) {
        batch_group_count = 1 : i64,
        dimension_numbers = {
          input_batch_dimension = 0 : i64,
          input_feature_dimension = 3 : i64,
          input_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>,
          kernel_input_feature_dimension = 2 : i64,
          kernel_output_feature_dimension = 3 : i64,
          kernel_spatial_dimensions = dense<[0, 1]> : tensor<2xi64>,
          output_batch_dimension = 0 : i64,
          output_feature_dimension = 3 : i64,
          output_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>},
        feature_group_count = 1 : i64,
        rhs_dilation = dense<1> : tensor<2xi64>,
        window_strides = dense<1> : tensor<2xi64>} : (tensor<1x4x5x2xf32>, tensor<3x2x2x1xf32>) -> tensor<1x2x3x1xf32>
  check.expect_almost_eq_const(%res, dense<[[
      [[1310.0],[1466.0],[1622.0]],
      [[2090.0],[2246.0],[2402.0]]
  ]]> : tensor<1x2x3x1xf32>) : tensor<1x2x3x1xf32>
  return
}

func @conv2d_1452x3221_same() attributes { iree.module.export } {
  %inputs = iree.unfoldable_constant dense<[[
      [[ 1.0,  2.0], [ 3.0,  4.0], [ 5.0,  6.0], [ 7.0,  8.0], [ 9.0, 10.0]],
      [[11.0, 12.0], [13.0, 14.0], [15.0, 16.0], [17.0, 18.0], [19.0, 20.0]],
      [[21.0, 22.0], [23.0, 24.0], [25.0, 26.0], [27.0, 28.0], [29.0, 30.0]],
      [[31.0, 32.0], [33.0, 34.0], [35.0, 36.0], [37.0, 38.0], [39.0, 40.0]]]]> : tensor<1x4x5x2xf32>
  %weights = iree.unfoldable_constant dense<[
      [[[ 1.0], [ 2.0]], [[ 3.0], [ 4.0]]],
      [[[ 5.0], [ 6.0]], [[ 7.0], [ 8.0]]],
      [[[ 9.0], [10.0]], [[11.0], [12.0]]]]> : tensor<3x2x2x1xf32>
  %res = "mhlo.convolution"(%inputs, %weights) {
       batch_group_count = 1 : i64,
       dimension_numbers = {
         input_batch_dimension = 0 : i64,
         input_feature_dimension = 3 : i64,
         input_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>,
         kernel_input_feature_dimension = 2 : i64,
         kernel_output_feature_dimension = 3 : i64,
         kernel_spatial_dimensions = dense<[0, 1]> : tensor<2xi64>,
         output_batch_dimension = 0 : i64,
         output_feature_dimension = 3 : i64,
         output_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>},
       feature_group_count = 1 : i64,
       padding = dense<[[1, 1], [0, 1]]> : tensor<2x2xi64>,
       rhs_dilation = dense<1> : tensor<2xi64>,
       window_strides = dense<1> : tensor<2xi64>} :
       (tensor<1x4x5x2xf32>, tensor<3x2x2x1xf32>) -> tensor<1x4x5x1xf32>
  check.expect_almost_eq_const(%res,  dense<[[
    [[ 600.0], [ 736.0], [ 872.0], [1008.0], [ 476.0]],
    [[1310.0], [1466.0], [1622.0], [1778.0], [ 805.0]],
    [[2090.0], [2246.0], [2402.0], [2558.0], [1135.0]],
    [[1080.0], [1152.0], [1224.0], [1296.0], [ 524.0]]]]> : tensor<1x4x5x1xf32>) : tensor<1x4x5x1xf32>
  return
}

func @conv2d_2451x2311_same() attributes { iree.module.export } {
  %inputs = iree.unfoldable_constant dense<[
      [[[ 1.0], [ 2.0], [ 3.0], [ 4.0], [ 5.0]],
       [[ 6.0], [ 7.0], [ 8.0], [ 9.0], [10.0]],
       [[11.0], [12.0], [13.0], [14.0], [15.0]],
       [[16.0], [17.0], [18.0], [19.0], [20.0]]],
      [[[21.0], [22.0], [23.0], [24.0], [25.0]],
       [[26.0], [27.0], [28.0], [29.0], [30.0]],
       [[31.0], [32.0], [33.0], [34.0], [35.0]],
       [[36.0], [37.0], [38.0], [39.0], [40.0]]]]> : tensor <2x4x5x1xf32>
  %weights = iree.unfoldable_constant dense<[
      [[[1.0]], [[2.0]], [[3.0]]],
      [[[4.0]], [[5.0]], [[6.0]]]]> : tensor <2x3x1x1xf32>
  %res = "mhlo.convolution"(%inputs, %weights) {
       batch_group_count = 1 : i64,
       dimension_numbers = {
         input_batch_dimension = 0 : i64,
         input_feature_dimension = 3 : i64,
         input_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>,
         kernel_input_feature_dimension = 2 : i64,
         kernel_output_feature_dimension = 3 : i64,
         kernel_spatial_dimensions = dense<[0, 1]> : tensor<2xi64>,
         output_batch_dimension = 0 : i64,
         output_feature_dimension = 3 : i64,
         output_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>},
       feature_group_count = 1 : i64,
       padding = dense<[[0, 1], [1, 1]]> : tensor<2x2xi64>,
       rhs_dilation = dense<1> : tensor<2xi64>,
       window_strides = dense<1> : tensor<2xi64>} :
       (tensor<2x4x5x1xf32>, tensor<2x3x1x1xf32>) -> tensor<2x4x5x1xf32>
  check.expect_almost_eq_const(%res, dense<[
    [[[ 80.0], [121.0], [142.0], [163.0], [100.0]],
     [[160.0], [226.0], [247.0], [268.0], [160.0]],
     [[240.0], [331.0], [352.0], [373.0], [220.0]],
     [[ 83.0], [104.0], [110.0], [116.0], [ 59.0]]],
    [[[400.0], [541.0], [562.0], [583.0], [340.0]],
     [[480.0], [646.0], [667.0], [688.0], [400.0]],
     [[560.0], [751.0], [772.0], [793.0], [460.0]],
     [[183.0], [224.0], [230.0], [236.0], [119.0]]]]> : tensor<2x4x5x1xf32>) : tensor<2x4x5x1xf32>
  return
}

func @conv2d_no_padding2() attributes { iree.module.export } {
  %inputs = iree.unfoldable_constant dense<[
       [[[  1.0,   2.0,   3.0],
         [  4.0,   5.0,   6.0],
         [  7.0,   8.0,   9.0],
         [ 10.0,  11.0,  12.0],
         [ 13.0,  14.0,  15.0]],
        [[ 16.0,  17.0,  18.0],
         [ 19.0,  20.0,  21.0],
         [ 22.0,  23.0,  24.0],
         [ 25.0,  26.0,  27.0],
         [ 28.0,  29.0,  30.0]],
        [[ 31.0,  32.0,  33.0],
         [ 34.0,  35.0,  36.0],
         [ 37.0,  38.0,  39.0],
         [ 40.0,  41.0,  42.0],
         [ 43.0,  44.0,  45.0]],
        [[ 46.0,  47.0,  48.0],
         [ 49.0,  50.0,  51.0],
         [ 52.0,  53.0,  54.0],
         [ 55.0,  56.0,  57.0],
         [ 58.0,  59.0,  60.0]]],
       [[[ 61.0,  62.0,  63.0],
         [ 64.0,  65.0,  66.0],
         [ 67.0,  68.0,  69.0],
         [ 70.0,  71.0,  72.0],
         [ 73.0,  74.0,  75.0]],
        [[ 76.0,  77.0,  78.0],
         [ 79.0,  80.0,  81.0],
         [ 82.0,  83.0,  84.0],
         [ 85.0,  86.0,  87.0],
         [ 88.0,  89.0,  90.0]],
        [[ 91.0,  92.0,  93.0],
         [ 94.0,  95.0,  96.0],
         [ 97.0,  98.0,  99.0],
         [100.0, 101.0, 102.0],
         [103.0, 104.0, 105.0]],
        [[106.0, 107.0, 108.0],
         [109.0, 110.0, 111.0],
         [112.0, 113.0, 114.0],
         [115.0, 116.0, 117.0],
         [118.0, 119.0, 120.0]]]]> : tensor<2x4x5x3xf32>
  %weights = iree.unfoldable_constant dense<[
      [[[  1.0,   2.0,   3.0,   4.0,   5.0,   6.0],
        [  7.0,   8.0,   9.0,  10.0,  11.0,  12.0],
        [ 13.0,  14.0,  15.0,  16.0,  17.0,  18.0]],
       [[ 19.0,  20.0,  21.0,  22.0,  23.0,  24.0],
        [ 25.0,  26.0,  27.0,  28.0,  29.0,  30.0],
        [ 31.0,  32.0,  33.0,  34.0,  35.0,  36.0]],
       [[ 37.0,  38.0,  39.0,  40.0,  41.0,  42.0],
        [ 43.0,  44.0,  45.0,  46.0,  47.0,  48.0],
        [ 49.0,  50.0,  51.0,  52.0,  53.0,  54.0]]],
      [[[ 55.0,  56.0,  57.0,  58.0,  59.0,  60.0],
        [ 61.0,  62.0,  63.0,  64.0,  65.0,  66.0],
        [ 67.0,  68.0,  69.0,  70.0,  71.0,  72.0]],
       [[ 73.0,  74.0,  75.0,  76.0,  77.0,  78.0],
        [ 79.0,  80.0,  81.0,  82.0,  83.0,  84.0],
        [ 85.0,  86.0,  87.0,  88.0,  89.0,  90.0]],
       [[ 91.0,  92.0,  93.0,  94.0,  95.0,  96.0],
        [ 97.0,  98.0,  99.0, 100.0, 101.0, 102.0],
        [103.0, 104.0, 105.0, 106.0, 107.0, 108.0]]]]> : tensor<2x3x3x6xf32>
  %res = "mhlo.convolution"(%inputs, %weights) {
       batch_group_count = 1 : i64,
       dimension_numbers = {
         input_batch_dimension = 0 : i64,
         input_feature_dimension = 3 : i64,
         input_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>,
         kernel_input_feature_dimension = 2 : i64,
         kernel_output_feature_dimension = 3 : i64,
         kernel_spatial_dimensions = dense<[0, 1]> : tensor<2xi64>,
         output_batch_dimension = 0 : i64,
         output_feature_dimension = 3 : i64,
         output_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>},
       feature_group_count = 1 : i64,
       rhs_dilation = dense<1> : tensor<2xi64>,
       window_strides = dense<1> : tensor<2xi64>} :
       (tensor<2x4x5x3xf32>, tensor<2x3x3x6xf32>) -> tensor<2x3x3x6xf32>
  check.expect_almost_eq_const(%res, dense<[
      [[[16065.0,  16290.0,  16515.0,  16740.0,  16965.0,  17190.0],
        [18873.0,  19152.0,  19431.0,  19710.0,  19989.0,  20268.0],
        [21681.0,  22014.0,  22347.0,  22680.0,  23013.0,  23346.0]],
       [[30105.0,  30600.0,  31095.0,  31590.0,  32085.0,  32580.0],
        [32913.0,  33462.0,  34011.0,  34560.0,  35109.0,  35658.0],
        [35721.0,  36324.0,  36927.0,  37530.0,  38133.0,  38736.0]],
       [[44145.0,  44910.0,  45675.0,  46440.0,  47205.0,  47970.0],
        [46953.0,  47772.0,  48591.0,  49410.0,  50229.0,  51048.0],
        [49761.0,  50634.0,  51507.0,  52380.0,  53253.0,  54126.0]]],
      [[[72225.0,  73530.0,  74835.0,  76140.0,  77445.0,  78750.0],
        [75033.0,  76392.0,  77751.0,  79110.0,  80469.0,  81828.0],
        [77841.0,  79254.0,  80667.0,  82080.0,  83493.0,  84906.0]],
       [[86265.0,  87840.0,  89415.0,  90990.0,  92565.0,  94140.0],
        [89073.0,  90702.0,  92331.0,  93960.0,  95589.0,  97218.0],
        [91881.0,  93564.0,  95247.0,  96930.0,  98613.0, 100296.0]],
       [[100305.0, 102150.0, 103995.0, 105840.0, 107685.0, 109530.0],
        [103113.0, 105012.0, 106911.0, 108810.0, 110709.0, 112608.0],
        [105921.0, 107874.0, 109827.0, 111780.0, 113733.0, 115686.0]]]]> : tensor<2x3x3x6xf32>) : tensor<2x3x3x6xf32>
  return
}
