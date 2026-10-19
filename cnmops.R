# Single-sample CNV calling; replacement implementation, not the missing author R file.
# Usage: Rscript cnmops.R BAM OUTPUT_DIRECTORY [WINDOW_BP]
args <- commandArgs(trailingOnly=TRUE)
if (length(args) < 2 || length(args) > 3) stop("Usage: cnmops.R BAM OUTDIR [WINDOW_BP]")
suppressPackageStartupMessages(library(cn.mops))
suppressPackageStartupMessages(library(Rsamtools))
bam <- args[1]
out <- args[2]
wl <- if (length(args) == 3) as.integer(args[3]) else 1000L
if (is.na(wl) || wl < 1) stop("WINDOW_BP must be positive")
if (!file.exists(bam)) stop("BAM does not exist")
dir.create(out, recursive=TRUE, showWarnings=FALSE)
sample <- "sample"
targets <- scanBamHeader(bam)[[1]]$targets
chromosomes <- names(targets)[sub("^chr", "", names(targets)) %in% as.character(1:22)]
if (length(chromosomes) != 22) stop("Expected all 22 autosomes")
raw_results <- list()
bed_results <- list()
for (chrom in chromosomes) {
    message(format(Sys.time()), " singlecn.mops: ", chrom, ", WL=", wl)
    counts <- getReadCountsFromBAM(BAMFiles=bam, sampleNames=sample,
                                  refSeqName=chrom, WL=wl, parallel=0)
    fit <- singlecn.mops(counts, parallel=0)
    fit <- calcIntegerCopyNumbers(fit)
    calls <- as.data.frame(cnvs(fit))
    write.table(calls, file.path(out, paste0(chrom, ".raw_calls.tsv")), sep="\t",
                quote=FALSE, row.names=FALSE, col.names=TRUE)
    if (nrow(calls)) {
        if (!"CN" %in% names(calls)) stop("Unexpected cn.mops output: missing CN")
        cn <- as.numeric(sub("^CN", "", as.character(calls$CN)))
        if (anyNA(cn)) stop("Unexpected copy-number values")
        keep <- cn != 2
        bed_results[[chrom]] <- data.frame(chrom=as.character(calls$seqnames[keep]),
            start=as.integer(calls$start[keep])-1L, end=as.integer(calls$end[keep]),
            type=ifelse(cn[keep] < 2, "DEL", "DUP"))
        raw_results[[chrom]] <- calls
    }
    rm(fit, counts)
    gc()
}
bed <- if (length(bed_results)) do.call(rbind, bed_results) else
    data.frame(chrom=character(), start=integer(), end=integer(), type=character())
write.table(bed, file.path(out, "cn.mops.author.bed"), sep="\t", quote=FALSE,
            row.names=FALSE, col.names=FALSE)
if (length(raw_results)) {
    write.table(do.call(rbind,raw_results), file.path(out,"all.raw_calls.tsv"),
                sep="\t",quote=FALSE,row.names=FALSE,col.names=TRUE)
}
writeLines(c(paste0("BAM=", bam), paste0("WINDOW_BP=", wl), "Scope=chr1-chr22",
             "Algorithm=singlecn.mops followed by calcIntegerCopyNumbers",
             "Baseline=CN2; CN<2 DEL; CN>2 DUP",
             "Raw coordinates=1-based closed; BED=0-based half-open",
             "Other singlecn.mops parameters=installed package defaults",
             "This replaces the unavailable author cnmops.R; original parameters unknown."),
           file.path(out,"protocol.txt"))
writeLines(capture.output(sessionInfo()),file.path(out,"sessionInfo.txt"))
message("Completed: ", nrow(bed), " CNV calls")
