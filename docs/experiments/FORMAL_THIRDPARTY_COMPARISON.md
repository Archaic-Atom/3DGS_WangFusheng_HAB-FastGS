# Formal unified third-party comparison

All quality metrics use the exact frozen FastGS PSNR/SSIM + VGG-LPIPS evaluator.

| scene | method | N | PSNR | SSIM | LPIPS | train s | FPS |
|---|---|---|---|---|---|---|---|
| truck | fastgs | 254,219 | 25.821 | 0.8772 | 0.1765 | 224.3 | 394.5 |
| truck | hab_main | 207,698 | 25.808 | 0.8739 | 0.1863 | 209.0 | 412.0 |
| truck | vanilla | 2,056,650 | 25.446 | 0.8847 | 0.1420 | 1367.9 | 110.7 |
| truck | speedy | 256,115 | 25.196 | 0.8678 | 0.1899 | 825.8 | 418.7 |
| truck | taming_matched | 207,693 | 24.847 | 0.8568 | 0.2018 | 321.4 | 222.1 |
| truck | mini | 323,656 | 25.268 | 0.8823 | 0.1382 | 1018.1 | 520.8 |
| truck | dash | 1,398,910 | 25.839 | 0.8846 | 0.1524 | 370.9 | 154.2 |
| truck | shorter | 2,584,064 | 25.225 | 0.8786 | 0.1424 | 310.5 | 199.2 |
| train | fastgs | 230,899 | 22.505 | 0.8081 | 0.2399 | 268.2 | 374.5 |
| train | hab_main | 189,337 | 22.427 | 0.8043 | 0.2470 | 247.4 | 390.4 |
| train | vanilla | 1,090,113 | 22.184 | 0.8221 | 0.1958 | 1102.3 | 119.9 |
| train | speedy | 107,152 | 21.715 | 0.7716 | 0.2904 | 662.8 | 488.8 |
| train | taming_matched | 189,335 | 21.759 | 0.7759 | 0.2815 | 333.2 | 235.8 |
| train | mini | 279,713 | 21.613 | 0.8113 | 0.2223 | 1034.9 | 552.1 |
| train | dash | 1,008,628 | 22.305 | 0.8180 | 0.2080 | 411.6 | 132.3 |
| train | shorter | 1,085,440 | 21.524 | 0.8126 | 0.1997 | 283.4 | 207.7 |
| playroom | fastgs | 181,184 | 30.616 | 0.9110 | 0.2614 | 204.1 | 436.5 |
| playroom | hab_main | 150,170 | 30.123 | 0.9091 | 0.2659 | 212.7 | 443.1 |
| playroom | vanilla | 1,849,357 | 30.138 | 0.9088 | 0.2407 | 1740.1 | 105.6 |
| playroom | speedy | 187,167 | 29.977 | 0.9061 | 0.2689 | 1025.0 | 461.8 |
| playroom | taming_matched | 150,163 | 29.893 | 0.9017 | 0.2799 | 392.7 | 217.3 |
| playroom | mini | 506,125 | 30.507 | 0.9136 | 0.2380 | 1324.3 | 455.3 |
| playroom | dash | 1,358,139 | 30.763 | 0.9115 | 0.2482 | 303.6 | 129.7 |
| playroom | shorter | 2,326,016 | 29.914 | 0.9008 | 0.2425 | 250.4 | 249.8 |
| drjohnson | fastgs | 253,655 | 29.431 | 0.9000 | 0.2721 | 283.4 | 378.3 |
| drjohnson | hab_main | 207,997 | 29.378 | 0.8996 | 0.2741 | 269.1 | 389.2 |
| drjohnson | vanilla | 3,115,339 | 29.389 | 0.9053 | 0.2360 | 2378.4 | 72.5 |
| drjohnson | speedy | 313,757 | 29.056 | 0.9001 | 0.2664 | 1252.2 | 386.4 |
| drjohnson | taming_matched | 207,989 | 29.276 | 0.8949 | 0.2850 | 410.1 | 204.3 |
| drjohnson | mini | 598,692 | 29.483 | 0.9070 | 0.2438 | 1498.9 | 412.3 |
| drjohnson | dash | 2,528,684 | 29.678 | 0.9052 | 0.2470 | 412.5 | 97.1 |
| drjohnson | shorter | 3,273,600 | 29.130 | 0.8943 | 0.2480 | 264.6 | 186.8 |

## HAB versus count-matched Taming

| scene | dFPS | dPSNR | dLPIPS | wall ratio | all gates |
|---|---|---|---|---|---|
| truck | +85.5% | +0.961 | -0.0155 | 0.650 | FAIL |
| train | +65.6% | +0.668 | -0.0345 | 0.742 | FAIL |
| playroom | +103.9% | +0.231 | -0.0140 | 0.542 | FAIL |
| drjohnson | +90.5% | +0.102 | -0.0109 | 0.656 | FAIL |

Broad same-count superiority gate (at least 3/4 scenes): **FAIL**.

Cross-rasterizer FPS uses each native renderer in an idle-GPU synchronized 30-pass process; it is descriptive, not same-process paired inference.

Wall is the complete official 30k training invocation; method-specific internal final evaluation behavior is disclosed and cross-method wall is not treated as exact paired inference.
