__precompile__(false)
module PIMC_Action_Interface


using PIMC_Common
using PIMC_Structs

using PIMC_Primitive_Action: init_action! as init_action_prim!
using PIMC_Primitive_Action: U as U_prim, K as K_prim, U_update as U_update_prim, U_stored as U_stored_prim
using PIMC_Primitive_Action: update_stored_slice_data as update_stored_slice_data_prim
using PIMC_Primitive_Action: init_stored as init_stored_prim
using PIMC_Primitive_Action: meas_E_th as meas_E_th_prim,  meas_E_vir as meas_E_vir_prim
using PIMC_Primitive_Action: meas_virial_pressure as meas_virial_pressure_prim
    
using PIMC_Chin_Action: init_action! as init_action_chin!
using PIMC_Chin_Action: update_stored_slice_data as update_stored_slice_data_chin, U as U_chin, K as K_chin
using PIMC_Chin_Action: U_update as U_update_chin, U_stored as U_stored_chin
using PIMC_Chin_Action: init_stored as init_stored_chin
using PIMC_Chin_Action: opt_chin_a1_chin_t0
using PIMC_Chin_Action: meas_E_th as meas_E_th_chin, meas_E_vir as meas_E_vir_chin
using PIMC_Chin_Action: meas_virial_pressure as meas_virial_pressure_chin

export init_action!, init_stored, update_stored_slice_data
export U, K, U_stored, U_update
export meas_E_th, meas_E_vir, meas_E_vir_loop, meas_virial_pressure
export get_ms, opt_chin_a1_chin_t0

# compile-time dispatch:
# ======================

# "Symmetric" physical slices where measurements are done
# needed for Chin Action, which has 3 slices per one τ 
# Compile-time dispatch


@inline function get_ms(::PrimitiveAction, M::Int64)
    return 1:M
end
@inline function get_ms(::ChinAction, M::Int64)
    return 2:3:M
end
@inline function get_ms(M::Int64)
    A = PIMC_Common.action
    return get_ms(A(), M)
end



@inline function init_action!(::PrimitiveAction, PIMC::t_pimc, beads::t_beads)
    return init_action_prim!(PIMC, beads)
end
@inline function init_action!(::ChinAction, PIMC::t_pimc, beads::t_beads)
    return init_action_chin!(PIMC, beads)
end
@inline function init_action!(PIMC::t_pimc, beads::t_beads)
    """user interface"""
    A = PIMC_Common.action
    return init_action!(A(), PIMC, beads)
end

@inline function init_stored(::PrimitiveAction, PIMC::t_pimc, beads::t_beads)
    return init_stored_prim(PIMC, beads)
end
@inline function init_stored(::ChinAction, PIMC::t_pimc, beads::t_beads)
    return init_stored_chin(PIMC, beads)
end
@inline function init_stored(PIMC::t_pimc, beads::t_beads)
    """user interface"""
    A = PIMC_Common.action
    return init_stored(A(), PIMC, beads)
end


function meas_E_th(::PrimitiveAction, PIMC::t_pimc, beads::t_beads, links::t_links, meas::t_measurement)
    return meas_E_th_prim(PIMC::t_pimc, beads::t_beads, links::t_links, meas::t_measurement)
end
function meas_E_th(::ChinAction, PIMC::t_pimc, beads::t_beads, links::t_links, meas::t_measurement)
    return meas_E_th_chin(PIMC::t_pimc, beads::t_beads, links::t_links, meas::t_measurement)
end
function meas_E_th(PIMC::t_pimc, beads::t_beads, links::t_links, meas::t_measurement)
    A = PIMC_Common.action  
    return meas_E_th(A(), PIMC, beads, links, meas)
end

function meas_E_vir(::PrimitiveAction, PIMC::t_pimc, beads::t_beads, links::t_links, meas::t_measurement; opt::Bool=false)
    return meas_E_vir_prim(PIMC::t_pimc, beads::t_beads, links::t_links, meas::t_measurement; opt=opt)
end
function meas_E_vir(::ChinAction, PIMC::t_pimc, beads::t_beads, links::t_links, meas::t_measurement; opt::Bool=false)
    return meas_E_vir_chin(PIMC::t_pimc, beads::t_beads, links::t_links, meas::t_measurement; opt=opt)
end
function meas_E_vir(PIMC::t_pimc, beads::t_beads, links::t_links, meas::t_measurement; opt::Bool=false)
    A = PIMC_Common.action  
    return meas_E_vir(A(), PIMC, beads, links, meas; opt=opt)
end

function meas_virial_pressure(::PrimitiveAction, PIMC::t_pimc, beads::t_beads, links::t_links, meas::t_measurement; opt::Bool=false)
    return meas_virial_pressure_prim(PIMC::t_pimc, beads::t_beads, links::t_links, meas::t_measurement; opt=opt)
end
function meas_virial_pressure(::ChinAction, PIMC::t_pimc, beads::t_beads, links::t_links, meas::t_measurement; opt::Bool=false)
    return meas_virial_pressure_chin(PIMC::t_pimc, beads::t_beads, links::t_links, meas::t_measurement; opt=opt)
end
function meas_virial_pressure(PIMC::t_pimc, beads::t_beads, links::t_links, meas::t_measurement; opt::Bool=false)
    A = PIMC_Common.action  
    return meas_virial_pressure(A(), PIMC, beads, links, meas; opt=opt)
end


@inline function U(::PrimitiveAction, PIMC::t_pimc, beads::t_beads) 
    return U_prim(PIMC, beads, id)
end
@inline function U(::ChinAction, PIMC::t_pimc, beads::t_beads)
    return U_chin(PIMC, beads)
end
@inline function U(PIMC::t_pimc, beads::t_beads)
    """user interface"""
    A = PIMC_Common.action
    return U(A(), PIMC, beads)
end

@inline function U_stored(::PrimitiveAction, PIMC::t_pimc, beads::t_beads, id::Int64)    
    return U_stored_prim(PIMC, beads, id)
end
@inline function U_stored(::ChinAction, PIMC::t_pimc, beads::t_beads, id::Int64)
    return U_stored_chin(PIMC, beads, id)
end
@inline function U_stored(PIMC::t_pimc, beads::t_beads, id::Int64)
    """user interface"""
    A = PIMC_Common.action
    return U_stored(A(), PIMC, beads, id)
end


@inline function U_update(::PrimitiveAction, PIMC::t_pimc, beads::t_beads,
                         Xold::AbstractArray{Float64}, id::Int64, act::Symbol,
                         fake::Bool)
    return U_update_prim(PIMC, beads, Xold, id, act, fake)
end
@inline function U_update(::ChinAction, PIMC::t_pimc, beads::t_beads,
                         Xold::AbstractArray{Float64}, id::Int64, act::Symbol,
                         fake::Bool)
    return U_update_chin(PIMC, beads, Xold, id, act, fake)
end

@inline function U_update(PIMC::t_pimc, beads::t_beads,
                         Xold::AbstractArray{Float64}, id::Int64, act::Symbol;
                          fake::Bool=false)
    # User interface
    A = PIMC_Common.action()  
    return U_update(A, PIMC, beads, Xold, id, act, fake)
end

@inline function K(::PrimitiveAction, PIMC::t_pimc, beads::t_beads, links::t_links, id::Int64)
    return K_prim(PIMC, beads, links, id)
end
@inline function K(::ChinAction, PIMC::t_pimc, beads::t_beads, links::t_links, id::Int64)
    return K_chin(PIMC, beads, links, id)
end
@inline function K(PIMC::t_pimc, beads::t_beads, links::t_links, id::Int64)
    """user interface"""
    A = PIMC_Common.action
    return K(A(), PIMC, beads, links, id)
end

@inline function update_stored_slice_data(::PrimitiveAction, PIMC::t_pimc, beads::t_beads, id::Int64)
    return update_stored_slice_data_prim(PIMC, beads, id)
end
@inline function update_stored_slice_data(::ChinAction, PIMC::t_pimc, beads::t_beads, id::Int64)
    return update_stored_slice_data_chin(PIMC, beads, id)
end
@inline function update_stored_slice_data(PIMC::t_pimc, beads::t_beads, id::Int64)
    """user interface"""
    A = PIMC_Common.action
    return update_stored_slice_data(A(), PIMC, beads, id)
end
# ==================================


end # module
