CXX      := g++
CXXFLAGS := -std=c++17 -O3 -march=native -mavx2 -mfma \
            -funroll-loops -fno-omit-frame-pointer \
            -I include
OMPFLAGS := -fopenmp

SRCDIR  := src
BINDIR  := bin
PLOTDIR := plots
RESDIR  := results

TARGETS := naive reordered unrolled blocked avx_vectorized \
           cache_aware register_kernel openmp_blocked warmup

.PHONY: all clean plots results dirs help

all: dirs $(TARGETS)

dirs:
	@mkdir -p $(BINDIR) $(PLOTDIR) $(RESDIR)

# --------------------------------------------------------------------------
# Per-target build rules
# --------------------------------------------------------------------------
naive: $(SRCDIR)/naive.cpp include/matrix.h
	$(CXX) $(CXXFLAGS) -o $(BINDIR)/$@ $<

reordered: $(SRCDIR)/reordered.cpp include/matrix.h
	$(CXX) $(CXXFLAGS) -o $(BINDIR)/$@ $<

unrolled: $(SRCDIR)/unrolled.cpp include/matrix.h
	$(CXX) $(CXXFLAGS) -o $(BINDIR)/$@ $<

blocked: $(SRCDIR)/blocked.cpp include/matrix.h
	$(CXX) $(CXXFLAGS) -o $(BINDIR)/$@ $<

avx_vectorized: $(SRCDIR)/avx_vectorized.cpp include/matrix.h
	$(CXX) $(CXXFLAGS) -o $(BINDIR)/$@ $<

cache_aware: $(SRCDIR)/cache_aware.cpp include/matrix.h
	$(CXX) $(CXXFLAGS) -o $(BINDIR)/$@ $<

register_kernel: $(SRCDIR)/register_kernel.cpp include/matrix.h
	$(CXX) $(CXXFLAGS) -o $(BINDIR)/$@ $<

openmp_blocked: $(SRCDIR)/openmp_blocked.cpp include/matrix.h
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) -o $(BINDIR)/$@ $<

warmup: $(SRCDIR)/warmup.cpp
	$(CXX) $(CXXFLAGS) -o $(BINDIR)/$@ $<

# --------------------------------------------------------------------------
# Run full benchmark suite (requires Linux perf)
# --------------------------------------------------------------------------
benchmark: all
	@bash scripts/benchmark.sh

# --------------------------------------------------------------------------
# Generate all plots from benchmark CSVs
# --------------------------------------------------------------------------
plots: $(RESDIR)
	python3 scripts/plot_results.py

# --------------------------------------------------------------------------
# Quick correctness check (small N, few iters)
# --------------------------------------------------------------------------
verify: all
	@bash scripts/verify.sh

clean:
	rm -rf $(BINDIR) $(PLOTDIR)/*.png

help:
	@echo ""
	@echo "Targets:"
	@echo "  all              Build all executables"
	@echo "  benchmark        Run full perf benchmark suite"
	@echo "  plots            Generate performance plots from CSVs"
	@echo "  verify           Quick correctness check"
	@echo "  clean            Remove binaries and plots"
	@echo ""
	@echo "Individual executables in bin/:"
	@echo "  naive            N I"
	@echo "  reordered        N I"
	@echo "  unrolled         N I"
	@echo "  blocked          N I B"
	@echo "  avx_vectorized   N I [B]"
	@echo "  cache_aware      N I"
	@echo "  register_kernel  N I"
	@echo "  openmp_blocked   N I [B]   (set OMP_NUM_THREADS)"
	@echo "  warmup           SIZE PATTERN ITERS [STRIDE]"
	@echo ""
