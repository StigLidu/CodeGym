import gymnasium
import ast
import json
from datetime import datetime, timedelta

class OptimalMeetingTimeEnv(gymnasium.Env):
    def __init__(self, env_str: str=None):
        super().__init__()

        # [Required] Define the action names
        self.OBSERVE = 0
        self.PARSE_TIME_INTERVAL = 1
        self.PARSE_SCHEDULE = 2
        self.GENERATE_POSSIBLE_START_TIMES = 3
        self.COUNT_CONFLICTS = 4
        self.FIND_BEST_START_TIME = 5
        self.DONE = 6

        # [Required] Define the action mapping
        self.func_mapping = {
            "Observe": self.OBSERVE,
            "ParseTimeInterval": self.PARSE_TIME_INTERVAL,
            "ParseSchedule": self.PARSE_SCHEDULE,
            "GeneratePossibleStartTimes": self.GENERATE_POSSIBLE_START_TIMES,
            "CountConflicts": self.COUNT_CONFLICTS,
            "FindBestStartTime": self.FIND_BEST_START_TIME,
            "Done": self.DONE
        }

        if env_str is not None:
            options = ast.literal_eval(env_str.split("@")[1])
            self.reset(options)
        else:
            self.reset()

    # [Required] Define the property and staticmethod of the environment
    @property
    def finished(self) -> bool:
        return self._done

    @property
    def reward(self):
        return float(self._reward)

    @staticmethod
    def from_env_str(env_str: str):
        prefix = "OptimalMeetingTimeEnv@"
        if not env_str.startswith(prefix):
            return None
        return OptimalMeetingTimeEnv(env_str=env_str)

    # [Required] Define the reset method of the environment
    def reset(self, options={}):
        self.duration = options.get("duration", 60)
        self.num_attendees = options.get("num_attendees", 0)
        self.attendees_schedules = options.get("attendees_schedules", [])
        self._reward = 0
        self._done = False
        self.step_count = 0
        return "Environment has been reset."

    # [Required] Get the reference answer of the environment
    def get_ref_answer(self):
        r"""
        Use the information in the environment to get the reference answer. 
        """
        start_of_day = datetime.strptime("09:00", "%H:%M")
        end_of_day = datetime.strptime("17:00", "%H:%M")
        duration_delta = timedelta(minutes=self.duration)

        all_intervals = []
        for schedule in self.attendees_schedules:
            intervals = schedule.split()
            parsed_intervals = []
            for interval in intervals:
                start_str, end_str = interval.split('-')
                start_time = datetime.strptime(start_str, "%H:%M")
                end_time = datetime.strptime(end_str, "%H:%M")
                parsed_intervals.append((start_time, end_time))
            all_intervals.extend(parsed_intervals)
        
        possible_start_times = []
        current_time = start_of_day
        while current_time + duration_delta <= end_of_day:
            possible_start_times.append(current_time)
            current_time += timedelta(minutes=1)
        
        min_conflicts = float('inf')
        best_start_time = None

        for start_time in possible_start_times:
            end_time = start_time + duration_delta
            conflicts = 0

            for interval in all_intervals:
                interval_start, interval_end = interval
                if start_time < interval_end and end_time > interval_start:
                    conflicts += 1
            
            if conflicts < min_conflicts:
                min_conflicts = conflicts
                best_start_time = start_time
        
        return best_start_time.strftime("%H:%M")

    # [Required] Define the step method of the environment
    def step(self, action: str):
        r"""
        Execute an action, and return the result information. 

        Args:
            action (str): The JSON string of the action name and parameters. 

        Returns:
            tuple[bool, str]: Whether the action is executed successfully, and the result information. 
        """
        self.step_count += 1
        try:
            call_dict = json.loads(action)
            assert "name" in call_dict, "function call doesn't have `name`"
            assert "parameters" in call_dict, "function call doesn't have `parameters`"
            action_name = call_dict["name"]
            params = call_dict["parameters"]

            if action_name not in self.func_mapping:
                raise ValueError(f"Invalid action: {action_name}")
            
            action_code = self.func_mapping[action_name]
            
            if action_code == self.OBSERVE:
                msg = self.Observe()
                
            elif action_code == self.PARSE_TIME_INTERVAL:
                if "interval_str" in params:
                    interval_str = params["interval_str"]
                    msg = self.ParseTimeInterval(interval_str)
                else:
                    msg = "Error: 'interval_str' parameter is missing for PARSE_TIME_INTERVAL action."
                    
            elif action_code == self.PARSE_SCHEDULE:
                if "schedule_str" in params:
                    schedule_str = params["schedule_str"]
                    msg = self.ParseSchedule(schedule_str)
                else:
                    msg = "Error: 'schedule_str' parameter is missing for PARSE_SCHEDULE action."
                    
            elif action_code == self.GENERATE_POSSIBLE_START_TIMES:
                if "duration" in params:
                    duration = params["duration"]
                    msg = self.GeneratePossibleStartTimes(duration)
                else:
                    msg = "Error: 'duration' parameter is missing for GENERATE_POSSIBLE_START_TIMES action."
                    
            elif action_code == self.COUNT_CONFLICTS:
                if "start_time_str" in params and "duration" in params and "intervals" in params:
                    start_time_str = params["start_time_str"]
                    duration = params["duration"]
                    intervals = params["intervals"]
                    msg = self.CountConflicts(start_time_str, duration, intervals)
                else:
                    msg = "Error: 'start_time_str', 'duration' or 'intervals' parameter is missing for COUNT_CONFLICTS action."
                    
            elif action_code == self.FIND_BEST_START_TIME:
                if "possible_start_times" in params and "duration" in params and "intervals" in params:
                    possible_start_times = params["possible_start_times"]
                    duration = params["duration"]
                    intervals = params["intervals"]
                    msg = self.FindBestStartTime(possible_start_times, duration, intervals)
                else:
                    msg = "Error: 'possible_start_times', 'duration' or 'intervals' parameter is missing for FIND_BEST_START_TIME action."
                    
            elif action_code == self.DONE:
                if "answer" in params:
                    answer = params["answer"]
                    msg = self.Done(answer)
                else:
                    msg = "Error: 'answer' parameter is missing for DONE action."
                    
        except Exception as e:
            msg = f"Error: {str(e)}"

        return True, msg

    # All the actions of the environment
    def Observe(self):
        r"""
    
        Obtain the current meeting duration, number of participants, and each participant's schedule from the current environment.
    
        Args:
            None
    
        Returns:
            str: A JSON string containing the meeting duration, number of participants, and each participant's schedule.
    
        Example Output:
            "{\"duration\": 60, \"num_attendees\": 3, \"attendees_schedules\": [\"09:00-10:00 14:00-15:30\", \"11:00-12:00 13:00-14:00\", \"10:00-11:00 15:00-16:00\"]}"
        """
        observation = {
            "duration": self.duration,
            "num_attendees": self.num_attendees,
            "attendees_schedules": self.attendees_schedules
        }
        return json.dumps(observation)

    def ParseTimeInterval(self, interval_str: str):
        r"""
    
        Parse a time interval string into timestamps of the start and end times.
    
        Args:
            interval_str (str): A time interval string in the format "hh:mm-hh:mm".
    
        Returns:
            str: A JSON string containing the start and end timestamps.
    
        Example Output:
            "{\"start_timestamp\": 32400, \"end_timestamp\": 36000}"
        """
        start_str, end_str = interval_str.split('-')
        start_time = datetime.strptime(start_str, "%H:%M")
        end_time = datetime.strptime(end_str, "%H:%M")
        
        start_timestamp = start_time.hour * 3600 + start_time.minute * 60
        end_timestamp = end_time.hour * 3600 + end_time.minute * 60
        
        result = {
            "start_timestamp": start_timestamp,
            "end_timestamp": end_timestamp
        }
        return json.dumps(result)

    def ParseSchedule(self, schedule_str: str):
        r"""
    
        Parse the entire schedule string and convert all time intervals within it into timestamp format.
    
        Args:
            schedule_str (str): A string containing multiple time intervals in "hh:mm-hh:mm" format.
    
        Returns:
            str: A JSON string containing all parsed time intervals.
    
        Example Output:
            "[{\"start_timestamp\": 32400, \"end_timestamp\": 36000}, {\"start_timestamp\": 50400, \"end_timestamp\": 55800}]"
        """
        intervals = schedule_str.split()
        parsed_intervals = []
        
        for interval in intervals:
            start_str, end_str = interval.split('-')
            start_time = datetime.strptime(start_str, "%H:%M")
            end_time = datetime.strptime(end_str, "%H:%M")
            
            start_timestamp = start_time.hour * 3600 + start_time.minute * 60
            end_timestamp = end_time.hour * 3600 + end_time.minute * 60
            
            parsed_intervals.append({
                "start_timestamp": start_timestamp,
                "end_timestamp": end_timestamp
            })
        
        return json.dumps(parsed_intervals)

    def GeneratePossibleStartTimes(self, duration: int):
        r"""
    
        Generate all possible meeting start times, starting from 09:00, with each possible start time at 1-minute intervals.
    
        Args:
            duration (int): Meeting duration (in minutes).
    
        Returns:
            str: A JSON string containing all possible start timestamps.
    
        Example Output:
            "[32400, 32460, 32520, ..., 57600]"
        """
        start_of_day = datetime.strptime("09:00", "%H:%M")
        end_of_day = datetime.strptime("17:00", "%H:%M")
        duration_delta = timedelta(minutes=duration)
        
        possible_start_times = []
        current_time = start_of_day
        
        while current_time + duration_delta <= end_of_day:
            timestamp = current_time.hour * 3600 + current_time.minute * 60
            possible_start_times.append(timestamp)
            current_time += timedelta(minutes=1)
        
        return json.dumps(possible_start_times)

    def CountConflicts(self, start_time_str: str, duration: int, intervals: list):
        r"""
    
        Calculate the number of conflicts between a meeting with a given start time and existing meetings.
    
        Args:
            start_time_str (str): String form of the meeting's start timestamp.
            duration (int): Meeting duration (in minutes).
            intervals (list): A list containing all existing meeting time intervals.
    
        Returns:
            str: String form of the conflict count.
    
        Example Output:
            "0"
        """
        start_timestamp = int(start_time_str)
        start_time = datetime.strptime("00:00", "%H:%M") + timedelta(seconds=start_timestamp)
        end_time = start_time + timedelta(minutes=duration)
        
        conflicts = 0
        
        for interval in intervals:
            interval_start = datetime.strptime("00:00", "%H:%M") + timedelta(seconds=interval["start_timestamp"])
            interval_end = datetime.strptime("00:00", "%H:%M") + timedelta(seconds=interval["end_timestamp"])
            
            if start_time < interval_end and end_time > interval_start:
                conflicts += 1
        
        return str(conflicts)

    def FindBestStartTime(self, possible_start_times: list, duration: int, intervals: list):
        r"""
    
        Find the best start time with the fewest conflicts from the possible start times; if there are multiple, select the earliest one.
    
        Args:
            possible_start_times (list): List of all possible start timestamps.
            duration (int): Meeting duration (in minutes).
            intervals (list): A list containing all existing meeting time intervals.
    
        Returns:
            str: The best start time in "hh:mm" format.
    
        Example Output:
            "12:00"
        """
        min_conflicts = float('inf')
        best_start_timestamp = None
        
        for start_timestamp in possible_start_times:
            start_time = datetime.strptime("00:00", "%H:%M") + timedelta(seconds=start_timestamp)
            end_time = start_time + timedelta(minutes=duration)
            
            conflicts = 0
            for interval in intervals:
                interval_start = datetime.strptime("00:00", "%H:%M") + timedelta(seconds=interval["start_timestamp"])
                interval_end = datetime.strptime("00:00", "%H:%M") + timedelta(seconds=interval["end_timestamp"])
                
                if start_time < interval_end and end_time > interval_start:
                    conflicts += 1
            
            if conflicts < min_conflicts:
                min_conflicts = conflicts
                best_start_timestamp = start_timestamp
        
        best_start_time = datetime.strptime("00:00", "%H:%M") + timedelta(seconds=best_start_timestamp)
        return best_start_time.strftime("%H:%M")

    def Done(self, answer):
        r"""
    
        Verify whether the final answer is correct and return the result information.
    
        Args:
            answer (str): The user-submitted best meeting start time in "hh:mm" format.
    
        Returns:
            str: Result information, including correctness and reward details.
    
        Example Output:
            "Your answer: 12:00, Reference answer: 12:00, Result: Correct, reward=1"
        """
        ref_answer = self.get_ref_answer()
        correct = answer == ref_answer
        self._reward = 1 if correct else 0
        self._done = True
        msg = f"Your answer: {answer}, Reference answer: {ref_answer}, Result: {'Correct' if correct else 'Incorrect'}"
        return msg + f", reward={self._reward}"

    # Define the solve method of the environment
    def solve(self):
        r"""
        Automatically call all actions to complete the complete process, and submit the answer for verification. 
    
        Returns:
            str: The result information of the final answer verification. 
        """
        import json
        
        observe_result = self.step(json.dumps({'name': 'Observe', 'parameters': {}}))[1]
        observe_data = json.loads(observe_result)
        duration = observe_data['duration']
        attendees_schedules = observe_data['attendees_schedules']
        
        all_intervals = []
        for schedule_str in attendees_schedules:
            parse_schedule_result = self.step(json.dumps({
                'name': 'ParseSchedule',
                'parameters': {'schedule_str': schedule_str}
            }))[1]
            intervals = json.loads(parse_schedule_result)
            all_intervals.extend(intervals)
        
        possible_starts_result = self.step(json.dumps({
            'name': 'GeneratePossibleStartTimes',
            'parameters': {'duration': duration}
        }))[1]
        possible_start_times = json.loads(possible_starts_result)
        
        best_start_result = self.step(json.dumps({
            'name': 'FindBestStartTime',
            'parameters': {
                'possible_start_times': possible_start_times,
                'duration': duration,
                'intervals': all_intervals
            }
        }))[1]
        
        return self.step(json.dumps({
            'name': 'Done',
            'parameters': {'answer': best_start_result}
        }))[1]
# Test the environment
if __name__ == "__main__":
    # test case 1 (from first example)
    print("Test Case 1:")
    env1 = OptimalMeetingTimeEnv.from_env_str(
        "OptimalMeetingTimeEnv@{"
        "\"duration\": 60, "
        "\"num_attendees\": 3, "
        "\"attendees_schedules\": ["
        "\"09:00-10:00 14:00-15:30\","
        "\"11:00-12:00 13:00-14:00\","
        "\"10:00-11:00 15:00-16:00\""
        "]}"
    )
    print(env1.solve())
    print("step count:", env1.step_count)
    
    # test case 2 (from second example)
    print("\nTest Case 2:")
    env2 = OptimalMeetingTimeEnv.from_env_str(
        "OptimalMeetingTimeEnv@{"
        "\"duration\": 30, "
        "\"num_attendees\": 2, "
        "\"attendees_schedules\": ["
        "\"13:00-14:00\","
        "\"10:00-11:30\""
        "]}"
    )
    print(env2.solve())
    print("step count:", env2.step_count)