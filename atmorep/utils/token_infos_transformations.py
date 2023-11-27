####################################################################################################
#
#  Copyright (C) 2022
#
####################################################################################################
#
#  project     : atmorep
#
#  author      : atmorep collaboration
# 
#  description :
#
#  license     :
#
####################################################################################################


####################################################################################################
def identity( args) :
  return args

####################################################################################################
def token_infos_transformation_year_1980_2015( token_infos) :
  
  if token_infos[0,0,0] in [2015.,2016.,2017.] :
    token_infos[:,:,0] -= 35
  elif token_infos[0,0,0] in [1980.,1981.,1982.] :
    token_infos[:,:,0] += 35
  else :
    assert False  # very basic sanity checking

  return token_infos

####################################################################################################
def token_infos_transformation_year_1979_2017( token_infos) :
  
  if token_infos[0,0,0] in [2017.,2018.,2019.,2020.,2021.,2022.] :
    token_infos[:,:,0] -= 38
  elif token_infos[0,0,0] in [1979.,1980.,1981.,1982.,1983.,1984.,1985.] :
    token_infos[:,:,0] += 38
  else :
    assert False  # very basic sanity checking

  return token_infos

####################################################################################################
def token_infos_transformation_year_1997_2015( token_infos) :
  
  if token_infos[0,0,0] in [2015.,2016.,2017.] :
    token_infos[:,:,0] -= 18
  elif token_infos[0,0,0] in [1997.,1998.,1999.] :
    token_infos[:,:,0] += 18
  else :
    assert False  # very basic sanity checking

  return token_infos

####################################################################################################
def token_infos_transformation_el_nino( token_infos) :
  
  # el nino: 01/2015
  # la nina: 01/2008

  if token_infos[0,0,0] in [2015.] :
    token_infos[:,:,0] -= 7
  elif token_infos[0,0,0] in [2008.] :
    token_infos[:,:,0] += 7

  # if token_infos[0,0,0] in [2015.] :
  #   token_infos[:,:,0] -= 31
  # elif token_infos[0,0,0] in [1984.] :
  #   token_infos[:,:,0] += 31
  # else :
  #   assert False  # very basic sanity checking

  return token_infos

####################################################################################################
def token_infos_transformation_extrapolation( token_infos) :
  
  if token_infos[0,0,0] in [2017.] :
    token_infos[:,:,0] += 5
  else :
    assert False  # very basic sanity checking

  return token_infos
