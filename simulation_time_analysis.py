import pstats

p = pstats.Stats('profile_output12345.prof')
p.sort_stats('cumulative').print_stats(10)