import random 

from CybORG.Agents import B_lineAgent
from CybORG.Agents.SimpleAgents.Meander import RedMeanderAgent
from CybORG.Shared import Observation
from CybORG.Shared.Actions import PrivilegeEscalate, ExploitRemoteService, Impact, \
    DiscoverNetworkServices, Sleep, DiscoverRemoteSystems 
        

class SleepyBLine(B_lineAgent):
    def get_action(self, observation, action_space):
        # print(self.action)
        """gets an action from the agent that should be performed based on the agent's internal state and provided observation and action space"""
        session = 0
        slept = False 

        while True:
            if observation['success'] == True:
                self.action += 1 if (self.action < 14 and not slept) else 0
            else:
                self.action = self.jumps[self.action]
            slept = False 

            if self.action in self.action_history:
                action = self.action_history[self.action]

            # Discover Remote Systems
            elif self.action == 0:
                self.initial_ip = observation['User0']['Interface'][0]['IP Address']
                self.last_subnet = observation['User0']['Interface'][0]['Subnet']

                if random.random() > 0.5:
                    action = Sleep()
                    slept = True 
                else:
                    action = DiscoverRemoteSystems(session=session, agent='Red', subnet=self.last_subnet)
            
            # Discover Network Services- new IP address found
            elif self.action == 1:
                hosts = [value for key, value in observation.items() if key != 'success']
                get_ip = lambda x : x['Interface'][0]['IP Address']
                interfaces = [get_ip(x) for x in hosts if get_ip(x)!= self.initial_ip]
                self.last_ip_address = random.choice(interfaces)
                action =DiscoverNetworkServices(session=session, agent='Red', ip_address=self.last_ip_address)

            # Exploit User1
            elif self.action == 2:
                 action = ExploitRemoteService(session=session, agent='Red', ip_address=self.last_ip_address)

            # Privilege escalation on User Host
            elif self.action == 3:
                hostname = [value for key, value in observation.items() if key != 'success' and 'System info' in value][0]['System info']['Hostname']
                action = PrivilegeEscalate(agent='Red', hostname=hostname, session=session)

            # Discover Network Services- new IP address found
            elif self.action == 4:
                self.enterprise_host = [x for x in observation if 'Enterprise' in x][0]
                self.last_ip_address = observation[self.enterprise_host]['Interface'][0]['IP Address']
                action = DiscoverNetworkServices(session=session, agent='Red', ip_address=self.last_ip_address)

            # Exploit- Enterprise Host
            elif self.action == 5:
                self.target_ip_address = [value for key, value in observation.items() if key != 'success'][0]['Interface'][0]['IP Address']
                action = ExploitRemoteService(session=session, agent='Red', ip_address=self.target_ip_address)

            # Privilege escalation on Enterprise Host
            elif self.action == 6:
                hostname = [value for key, value in observation.items() if key != 'success' and 'System info' in value][0]['System info']['Hostname']
                action = PrivilegeEscalate(agent='Red', hostname=hostname, session=session)

            # Scanning the new subnet found.
            elif self.action == 7:
                self.last_subnet = observation[self.enterprise_host]['Interface'][0]['Subnet']
                action = DiscoverRemoteSystems(subnet=self.last_subnet, agent='Red', session=session)

            # Discover Network Services- Enterprise2
            elif self.action == 8:
                self.target_ip_address = [value for key, value in observation.items() if key != 'success'][2]['Interface'][0]['IP Address']
                action = DiscoverNetworkServices(session=session, agent='Red', ip_address=self.target_ip_address)

            # Exploit- Enterprise2
            elif self.action == 9:
                self.target_ip_address = [value for key, value in observation.items() if key != 'success'][0]['Interface'][0]['IP Address']
                action = ExploitRemoteService(session=session, agent='Red', ip_address=self.target_ip_address)

            # Privilege escalation on Enterprise2
            elif self.action == 10:
                hostname = [value for key, value in observation.items() if key != 'success' and 'System info' in value][0]['System info']['Hostname']
                action = PrivilegeEscalate(agent='Red', hostname=hostname, session=session)

            # Discover Network Services- Op_Server0
            elif self.action == 11:
                action = DiscoverNetworkServices(session=session, agent='Red', ip_address=observation['Op_Server0']['Interface'][0]['IP Address'])

            # Exploit- Op_Server0
            elif self.action == 12:
                info = [value for key, value in observation.items() if key != 'success']
                if len(info) > 0:
                    action = ExploitRemoteService(agent='Red', session=session, ip_address=info[0]['Interface'][0]['IP Address'])
                else:
                    self.action = 0
                    continue
            # Privilege escalation on Op_Server0
            elif self.action == 13:
                action = PrivilegeEscalate(agent='Red', hostname='Op_Server0', session=session)
            # Impact on Op_server0
            elif self.action == 14:
                action = Impact(agent='Red', session=session, hostname='Op_Server0')

            if self.action not in self.action_history:
                self.action_history[self.action] = action
            return action
        
class SleepyMeander(RedMeanderAgent):
    def get_action(self, observation, action_space):
        """gets an action from the agent that should be performed based on the agent's internal state and provided observation and action space"""
        self._process_success(observation)

        session = list(action_space['session'].keys())[0]
        
        # Always impact if able
        if 'Op_Server0' in self.escalated_hosts:
            self.last_host = 'Op_Server0'
            return Impact(agent='Red', hostname='Op_Server0', session=session)

        # start by scanning
        for subnet in action_space["subnet"]:
            if not action_space["subnet"][subnet] or subnet in self.scanned_subnets:
                continue

            if random.random() > 0.5:
                self.scanned_subnets.append(subnet)
                return DiscoverRemoteSystems(subnet=subnet, agent='Red', session=session)
            else:
                return Sleep()
            
        # discover network services
        # # act on ip addresses discovered in first subnet
        addresses = [i for i in action_space["ip_address"]]
        random.shuffle(addresses)
        
        for address in addresses:
            if not action_space["ip_address"][address] or address in self.scanned_ips:
                continue
            self.scanned_ips.append(address)

            return DiscoverNetworkServices(ip_address=address, agent='Red', session=session)
        # priv esc on owned hosts
        hostnames = [x for x in action_space['hostname'].keys()]
        random.shuffle(hostnames)
        for hostname in hostnames:
            # test if host is not known
            if not action_space["hostname"][hostname]:
                continue
            # test if host is already priv esc
            if hostname in self.escalated_hosts:
                continue
            # test if host is exploited
            if hostname in self.host_ip_map and self.host_ip_map[hostname] not in self.exploited_ips:
                continue
            self.escalated_hosts.append(hostname)
            self.last_host = hostname
            return PrivilegeEscalate(hostname=hostname, agent='Red', session=session)

        # access unexploited hosts
        for address in addresses:
            # test if output of observation matches expected output
            if not action_space["ip_address"][address] or address in self.exploited_ips:
                continue
            self.exploited_ips.append(address)
            self.last_ip = address
            return ExploitRemoteService(ip_address=address, agent='Red', session=session)

        raise NotImplementedError('Red Meander has run out of options!')