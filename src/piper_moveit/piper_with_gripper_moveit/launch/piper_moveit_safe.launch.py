# piper_moveit_safe.launch.py

from pathlib import Path
import yaml

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, RegisterEventHandler, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.actions import Node
from launch.conditions import IfCondition, UnlessCondition
from launch.event_handlers import OnProcessStart

from moveit_configs_utils import MoveItConfigsBuilder
from moveit_configs_utils.launch_utils import add_debuggable_node, DeclareBooleanLaunchArg


# ---------- helpers ----------
def _load_yaml(path: Path):
    try:
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return data if data is not None else {}
    except Exception:
        return {}

def _coerce_joint_limits_to_float(cfg: dict) -> dict:
    if not cfg:
        return cfg
    if "default_velocity_scaling_factor" in cfg:
        cfg["default_velocity_scaling_factor"] = float(cfg["default_velocity_scaling_factor"])
    if "default_acceleration_scaling_factor" in cfg:
        cfg["default_acceleration_scaling_factor"] = float(cfg["default_acceleration_scaling_factor"])
    jl = cfg.get("joint_limits", {})
    for _, v in jl.items():
        if "max_velocity" in v:
            v["max_velocity"] = float(v["max_velocity"])
        if "max_acceleration" in v:
            v["max_acceleration"] = float(v["max_acceleration"])
        if "has_velocity_limits" in v:
            v["has_velocity_limits"] = bool(v["has_velocity_limits"])
        if "has_acceleration_limits" in v:
            v["has_acceleration_limits"] = bool(v["has_acceleration_limits"])
    return cfg

def _coerce_cartesian_limits_to_float(cfg: dict) -> dict:
    if not cfg:
        return cfg
    cl = cfg.get("cartesian_limits", {})
    if not cl:
        # If cartesian_limits is missing or empty, ensure it exists as a dict
        cfg["cartesian_limits"] = {}
        cl = cfg["cartesian_limits"]
    for k in ("max_trans_vel", "max_trans_acc", "max_trans_dec", "max_rot_vel"):
        if k in cl:
            try:
                cl[k] = float(cl[k])
            except (ValueError, TypeError):
                # If conversion fails, keep original value (will cause error but at least won't crash)
                pass
    return cfg


# ---------- launch entry ----------
def generate_launch_description():
    moveit_config = MoveItConfigsBuilder(
        "piper", package_name="piper_with_gripper_moveit"
    ).to_moveit_configs()

    ld = LaunchDescription()

    # Declare use_sim_time early since it's used by multiple nodes
    ld.add_action(DeclareBooleanLaunchArg("use_sim_time", default_value=True))  # False for real robot

    # Publish TF from the URDF (needed for real robot)
    ld.add_action(
        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            parameters=[
                moveit_config.robot_description,
                {"use_sim_time": LaunchConfiguration("use_sim_time")},
            ],
            output="screen",
        )
    )

    # Static transforms for virtual joints (world <-> base) like demo.launch.py provides
    ld.add_action(
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                str(moveit_config.package_path / "launch/static_virtual_joint_tfs.launch.py")
            ),
        )
    )

    _add_move_group_safe(ld, moveit_config)
    _add_rviz_safe(ld, moveit_config)

    # Only spawn ros2_control + controllers for real robot.
    # In simulation (use_sim_time=True), Gazebo already provides them.
    _add_ros2_control_real_only(ld, moveit_config)

    return ld


def _add_move_group_safe(ld: LaunchDescription, moveit_config):
    # launch args
    ld.add_action(DeclareBooleanLaunchArg("debug", default_value=False))
    ld.add_action(DeclareBooleanLaunchArg("allow_trajectory_execution", default_value=True))
    ld.add_action(DeclareBooleanLaunchArg("publish_monitored_planning_scene", default_value=True))
    ld.add_action(DeclareBooleanLaunchArg("monitor_dynamics", default_value=False))
    ld.add_action(DeclareLaunchArgument("capabilities", default_value=""))
    ld.add_action(DeclareLaunchArgument("disable_capabilities", default_value=""))

    should_publish = LaunchConfiguration("publish_monitored_planning_scene")

    move_group_configuration = {
        "publish_robot_description_semantic": True,
        "allow_trajectory_execution": LaunchConfiguration("allow_trajectory_execution"),
        "capabilities": ParameterValue(LaunchConfiguration("capabilities"), value_type=str),
        "disable_capabilities": ParameterValue(LaunchConfiguration("disable_capabilities"), value_type=str),
        "publish_planning_scene": should_publish,
        "publish_geometry_updates": should_publish,
        "publish_state_updates": should_publish,
        "publish_transforms_updates": should_publish,
        "monitor_dynamics": False,
    }

    # Start from base dict but drop any bundled joint-limits so we own robot_description_planning
    base_params = moveit_config.to_dict()
    base_params.pop("robot_description_planning", None)

    # Safe overlays
    pkg_path = Path(moveit_config.package_path)
    safe_cfg = pkg_path / "config" / "config_safe"

    kinematics      = _load_yaml(safe_cfg / "kinematics_kdl_safe.yaml")
    joint_limits    = _coerce_joint_limits_to_float(_load_yaml(safe_cfg / "joint_limits_safe.yaml"))
    pilz_limits     = _coerce_cartesian_limits_to_float(_load_yaml(safe_cfg / "pilz_cartesian_limits_safe.yaml"))
    planning_pipes  = _load_yaml(safe_cfg / "planning_pipelines_ompl_pilz.yaml")
    ompl_cfg        = _load_yaml(safe_cfg / "ompl_planning_safe.yaml")
    chomp_cfg       = _load_yaml(safe_cfg / "chomp_planning_safe.yaml")
    controllers     = _load_yaml(safe_cfg / "moveit_controllers_safe.yaml")
    traj_exec       = _load_yaml(safe_cfg / "trajectory_execution_tuned.yaml")
    sensors3d       = _load_yaml(safe_cfg / "sensors_3d_lidar.yaml")  # Load lidar sensor config for octomap

    # Merge joint + cartesian limits into robot_description_planning (Pilz reads it here)
    robot_description_planning = {}
    if joint_limits:
        robot_description_planning.update(joint_limits)
    if pilz_limits:
        # pilz_limits contains "cartesian_limits" key, merge it properly
        robot_description_planning.update(pilz_limits)

    move_group_params = [
        base_params,
        move_group_configuration,
        {"use_sim_time": LaunchConfiguration("use_sim_time")},
    ]

    if robot_description_planning:
        move_group_params.append({"robot_description_planning": robot_description_planning})
    if kinematics:
        move_group_params.append({"robot_description_kinematics": kinematics})
    if planning_pipes:
        move_group_params.append({"planning_pipelines": planning_pipes})
    if ompl_cfg:
        move_group_params.append({"ompl": ompl_cfg})
    if chomp_cfg:
        move_group_params.append({"chomp": chomp_cfg})
    if controllers:
        move_group_params.append(controllers)
    if traj_exec:
        move_group_params.append(traj_exec)
    if sensors3d:
        move_group_params.append(sensors3d)


    add_debuggable_node(
        ld,
        package="moveit_ros_move_group",
        executable="move_group",
        commands_file=str(pkg_path / "launch" / "gdb_settings.gdb"),
        output="screen",
        parameters=move_group_params,
        extra_debug_args=["--debug"],
        additional_env={"DISPLAY": ":0"},
    )


def _add_rviz_safe(ld: LaunchDescription, moveit_config):
    ld.add_action(DeclareBooleanLaunchArg("debug", default_value=False))
    ld.add_action(
        DeclareLaunchArgument(
            "rviz_config",
            default_value=str(moveit_config.package_path / "config/moveit.rviz"),
        )
    )

    pkg_path = Path(moveit_config.package_path)
    safe_cfg = pkg_path / "config" / "config_safe"

    kinematics     = _load_yaml(safe_cfg / "kinematics_kdl_safe.yaml")
    planning_pipes = _load_yaml(safe_cfg / "planning_pipelines_ompl_pilz.yaml")
    ompl_cfg       = _load_yaml(safe_cfg / "ompl_planning_safe.yaml")

    # IMPORTANT: Give RViz the robot model
    rviz_parameters = [
        {"use_sim_time": LaunchConfiguration("use_sim_time")},
        moveit_config.robot_description,
        moveit_config.robot_description_semantic,
    ]

    if planning_pipes:
        rviz_parameters.append({"planning_pipelines": planning_pipes})
    else:
        rviz_parameters.append(moveit_config.planning_pipelines)

    if kinematics:
        rviz_parameters.append({"robot_description_kinematics": kinematics})
    else:
        rviz_parameters.append(moveit_config.robot_description_kinematics)

    if ompl_cfg:
        rviz_parameters.append({"ompl": ompl_cfg})

    add_debuggable_node(
        ld,
        package="rviz2",
        executable="rviz2",
        output="screen",
        respawn=False,
        arguments=["-d", LaunchConfiguration("rviz_config")],
        parameters=rviz_parameters,
    )


def _add_ros2_control_real_only(ld: LaunchDescription, moveit_config):
    """Add ros2_control + controllers only when NOT in simulation.
    Gazebo already provides its own controller_manager and spawns controllers,
    so launching a second one causes 'already loaded' errors and CONTROL_FAILED."""
    not_sim = UnlessCondition(LaunchConfiguration("use_sim_time"))

    cm_node = Node(
        package="controller_manager",
        executable="ros2_control_node",
        parameters=[
            moveit_config.robot_description,
            {"use_sim_time": LaunchConfiguration("use_sim_time")},
            str(moveit_config.package_path / "config/ros2_controllers.yaml"),
        ],
        output="screen",
        condition=not_sim,
    )
    ld.add_action(cm_node)

    spawn_include = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            str(moveit_config.package_path / "launch/spawn_controllers.launch.py")
        ),
        condition=not_sim,
    )
    ld.add_action(
        RegisterEventHandler(
            OnProcessStart(
                target_action=cm_node,
                on_start=[TimerAction(period=2.0, actions=[spawn_include])],
            )
        )
    )
