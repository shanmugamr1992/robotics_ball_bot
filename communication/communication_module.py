class CommunicationModule:
    
    @classmethod
    def get_module(cls, run_on_nano=True, ip=None, port=None):
        if run_on_nano:
            from .local_communication import LocalCommunication
            return LocalCommunication()
        else:
            from .lan_communication import LanCommunication
            return LanCommunication(ip, port)
