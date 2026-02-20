# Rethinking the Functionality of Latent Representation: A Logarithmic Rate-Distortion Model for Learned Image Compression (TCSVT2025)


Please feel free to contact Ziqing Ge (ziqing.ge@vipl.ict.ac.cn) if you have any questions.

## Abstract
End-to-end optimized Learned Image Compression (LIC) has demonstrated remarkable performance in terms of Rate-Distortion (R-D) efficiency. However, the R-D characteristics of LIC codecs remain underexplored. Previous research has attempted to investigate the R-D behavior through numerical and statistical approaches, but these methods often provide only empirical results, lacking theoretical insights. In this work, we introduce a novel methodology for studying the R-D characteristics of LIC. By rethinking the LIC paradigm from a fresh perspective, we propose a plug-and-play module, the Latent-domain Auto-Encoder (LAE). This innovative approach not only naturally leads to Variable Bit-Rate (VBR) compression, but also allows for a theoretical modeling of the R-D behavior of LIC codecs.  Our findings reveal that the bit-rate is the logarithmic sum of the neurons $n_\lambda$ in our designed network's last layer, plus a constant $C$ introduced by image content, formally expressed as $R_\lambda = \sum \log n_\lambda + C$. This insight is pivotal, as it underscores how the bit-rate can be systematically derived from the latent representations. Further analysis demonstrates that our proposed $R-\lambda$ model enables effective rate control for learned image codecs, enhancing their adaptability and accuracy. Experimental results validate that our VBR method surpasses fixed-rate coding by 2.9% in terms of BD-rate. Additionally, the proposed $R-\lambda$ model exhibits superior rate control performance, suggesting that it not only elucidates the underlying R-D characteristics of LIC but also significantly enhances its practical deployment in real-world applications. 

## Model
`models/tcm.py` includes the model, where TCM_VBR is our proposed VBR method integrated on Liu2023 [46]. In this file, the module fcn represents the $f_\lambda$ function which embeds $\lambda$, mask is a module to generate $m$ in Eq. (12), and LAE is the LAE module as introduced in the paper. 

## Training
`train_latent_e2e.py` is a toy experiment to explore the relationship between latents of different $\lambda$, which is not included in the original paper.

`train_vbr.py` is the main entry of training, where you may modify some configurations, e.g. dataset path.


## Weights and Testing
The pretrained weight is available at `https://drive.google.com/file/d/13A88JG5rzH5BHn1TEn6GhCfyrfahgQ_L/view?usp=drive_link`. 
To evaluate this model, downloat it and put it under `vbr_models/`, then run

`python online_training_vbr.py -l [0~5] -d [Your dataset directory]`.

## Notes
This implementation is not original codes of our TCSVT2025 paper, because are rearranged by us. This repo is a re-implementation, but the core codes are almost the same and results are also consistent with original results. 



