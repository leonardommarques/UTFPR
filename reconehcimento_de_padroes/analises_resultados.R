# -------------------------------------------- #
# -- Dynamic Selection fo Classifiers (DSC) -- #
# -- analysis -- #
# -------------------------------------------- #

require(tidyverse)
require(plyr)
require(ggplot2)
require(data.table)


`%+%` = function(a,b) paste0(a,b)


folder_ = "/Users/leo/Documents/Estudos/UTFPR/reconehcimento_de_padroes/trabalho_final/simulations/"

sim_df = read.csv(folder_ %+% 'all_simulations.csv', sep=',', dec='.') %>%
  filter(strategy != '') %>%
  mutate(strategy = gsub(strategy, patt='ds_',repl= '')) %>%
  mutate(strategy = gsub(strategy, patt='_', repl=' ')) %>%
  mutate(strategy = toupper(strategy))


sim_df %>% names()
sim_df %>%
  group_by(strategy) %>%
  dplyr::summarise(accuracy_mean = mean(accuracy)) 

# -------------------------- #
# -- strategies accuracy
# -------------------------- #
acc_summary = sim_df %>%
  group_by(strategy) %>%
  dplyr::summarise(
    accuracy_mean = mean(accuracy)
    , accuracy_lcl = quantile(accuracy, 0.5/2)
    , accuracy_ucl = quantile(accuracy, 1-0.5/2)
    ) %>%
  arrange(-accuracy_mean)

# -- mean accuracy by strategy
gg_mean_acc_by_strategy = acc_summary %>%
  mutate(strategy =reorder(strategy,-accuracy_mean)) %>%
  ggplot(aes(x = strategy, y = accuracy_mean, fill=strategy))+
  geom_bar(sta = 'identity') +
  geom_errorbar(aes(ymin = accuracy_lcl, ymax = accuracy_ucl), width=.2) +
  labs(title = 'Acurácia Média'
       , subtitle = 'Intervalo de confiança empírico de 95%'
       , x = 'Método'
       , y = 'Acurácia') +
  scale_y_continuous(breaks = 0:10/10) +
  coord_cartesian(ylim = c(0,1)) +
  theme(legend.position = 'none')

gg_mean_acc_by_strategy

# -- mean acc by k -- #
acc_k_strat_summ = sim_df %>%
  group_by(strategy, k) %>%
  dplyr::summarise(
    accuracy_mean = mean(accuracy)
    , accuracy_ucl = quantile(accuracy, 1-0.5/2)
    , accuracy_lcl = quantile(accuracy, 0.5/2)
  )

# add fake K values for single best (SB)
acc_k_strat_summ_SB = acc_k_strat_summ %>%
  filter(strategy == 'SINGLE BEST')
acc_k_strat_summ_SB = rbind(acc_k_strat_summ_SB, acc_k_strat_summ_SB)
acc_k_strat_summ_SB$k = c(1,300)

acc_k_strat_summ = acc_k_strat_summ %>%
  filter(strategy != 'SINGLE BEST') %>%
  rbind(acc_k_strat_summ_SB)


gg_mean_acc_by_k = acc_k_strat_summ %>%
  ggplot(aes(x = k, y = accuracy_mean, col = strategy)) + 
  geom_line() +
  labs(title = 'Acurácia Média'
       # , subtitle = 'Intervalo de confiança empírico de 95%'
       , x = 'K'
       , y = 'Acurácia'
       , col='Método: ') +
  scale_x_continuous(breaks = function(limits) pretty(limits, 20)) +
  coord_cartesian(ylim = c(0.5,.75)) +
  theme(legend.position = 'bottom')

gg_mean_acc_by_k

gg_mean_acc_by_k_sep = gg_mean_acc_by_k + 
  geom_ribbon(
    aes(ymin = accuracy_lcl
        , ymax = accuracy_ucl
        , fill = strategy)
    , alpha = 0.3
    , col = "white") + 
  labs(title = 'Acurácia Média'
       , subtitle = 'Intervalo de confiança empírico de 95%'
       , x = 'K'
       , y = 'Acurácia'
       , col = 'Método'
       , fill = 'Método') +
  coord_cartesian(ylim = c(0,1)) +
  # geom_point() +
  facet_wrap(~strategy, ncol = 1)

gg_mean_acc_by_k_sep

# -- con escessão do LA OLA , os métodos matém a mesma variabildade de acurácia 
# para diferentes valores de k

# -- tamanho do data set
names(sim_df)
acc_n_strat_summ = sim_df %>%
  group_by(strategy, n_samples_) %>%
  dplyr::summarise(
    accuracy_mean = mean(accuracy)
    , accuracy_lcl = quantile(accuracy, 0.5/2)
    , accuracy_ucl = quantile(accuracy, 1-0.5/2)
  )

acc_n_strat_summ %>%
  ggplot(aes(x = n_samples_, y = accuracy_mean, col = strategy)) + 
  # geom_line() +
  geom_point() +
  labs(title = 'Acurácia Média'
       # , subtitle = 'Intervalo de confiança empírico de 95%'
       , x = 'K'
       , y = 'Acurácia'
       , col='Método: ') +
  scale_x_continuous(breaks = function(limits) pretty(limits, 20)) +
  theme(legend.position = 'bottom')




resutls = list(
  gg_mean_acc_by_strategy = gg_mean_acc_by_strategy
  , gg_mean_acc_by_k = gg_mean_acc_by_k
  , gg_mean_acc_by_k_sep = gg_mean_acc_by_k_sep)


resutls %>% 
  names() %>%
  llply(function(gg_, aux_list = resutls){
    ggsave(
      folder_ %+% gg_ %+% '.jpg'
      , aux_list[[gg_]]
    )
  })

ggsave(folder_ %+% 'urdur.jpg'
       , resutls[[1]])




