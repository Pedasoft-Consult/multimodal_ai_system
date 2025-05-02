# src/ingestion/kafka_consumer.py
from kafka import KafkaConsumer
import json
import threading
import time


class KafkaProcessor:
    def __init__(self, bootstrap_servers, topics, group_id="multimodal-processor"):
        self.consumer = KafkaConsumer(
            *topics,
            bootstrap_servers=bootstrap_servers,
            group_id=group_id,
            auto_offset_reset='latest',
            value_deserializer=lambda m: json.loads(m.decode('utf-8'))
        )

        self.running = False
        self.processor_thread = None
        self.processors = {}

    def register_processor(self, topic, processor_func):
        """Register a function to process messages from a specific topic"""
        self.processors[topic] = processor_func

    def process_message(self, message):
        """Process a message from Kafka"""
        topic = message.topic
        value = message.value

        # Call the registered processor function for this topic
        if topic in self.processors:
            try:
                self.processors[topic](value)
            except Exception as e:
                print(f"Error processing message from {topic}: {e}")
        else:
            print(f"No processor registered for topic {topic}")

    def start(self):
        """Start processing messages"""
        if self.running:
            return

        self.running = True
        self.processor_thread = threading.Thread(target=self._process_loop)
        self.processor_thread.daemon = True
        self.processor_thread.start()

    def stop(self):
        """Stop processing messages"""
        self.running = False
        if self.processor_thread:
            self.processor_thread.join(timeout=5.0)

    def _process_loop(self):
        """Main processing loop"""
        try:
            while self.running:
                message_batch = self.consumer.poll(timeout_ms=1000)

                for topic_partition, messages in message_batch.items():
                    for message in messages:
                        self.process_message(message)

        finally:
            self.consumer.close()